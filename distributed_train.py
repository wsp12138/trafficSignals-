import copy
import csv
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path

import numpy as np
import torch

from common.arguments import get_common_args
from main import (
    ROOT_DIR,
    align_signal_obs,
    build_cav_vehicle_set,
    build_signal_action_masks,
    build_signal_agent,
    build_vehicle_agent,
    is_vehicle_decision_finished,
    resolve_dataset,
    write_episode_logs,
)
from vehicle_single_env_finally import Vehicle_single_env


os.environ["LIBSUMO_AS_TRACI"] = "1"


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def make_weight_packet(signal_agent, vehicle_agent):
    return {
        "signal_policy": cpu_state_dict(signal_agent.policy_net),
        "vehicle_policy": cpu_state_dict(vehicle_agent.policy_net),
    }


def load_weight_packet(signal_agent, vehicle_agent, weights):
    signal_agent.policy_net.load_state_dict(weights["signal_policy"])
    vehicle_agent.policy_net.load_state_dict(weights["vehicle_policy"])


def make_vehicle_transition(exp, next_state, next_mask, done_flag):
    clipped_reward = np.clip(exp["acc_reward"], -50.0, 250.0)
    return (
        (exp["state"], exp["mask"]),
        exp["action"],
        clipped_reward,
        (next_state, next_mask),
        done_flag,
        max(exp["tau"], 1),
    )


def accumulate_vehicle_reward(agent, exp, step_reward):
    tau_clip = float(getattr(agent.args, "tau_discount_clip", 10.0))
    reward_tau = min(float(exp["tau"]), tau_clip)
    exp["acc_reward"] += (agent.gamma ** reward_tau) * float(step_reward)
    exp["tau"] += 1


def create_env(args, net_file, route_file, cav_vehicle_ids, enable_vehicle_routing):
    env = Vehicle_single_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=args.num_seconds,
        fixed_ts=False,
    )
    env.set_controlled_vehicle_ids(cav_vehicle_ids if enable_vehicle_routing else set())
    return env


def run_rollout_episode(
    args,
    env,
    agent_names,
    signal_agent,
    vehicle_agent,
    max_signal_obs_dim,
    max_signal_action_dim,
    episode_idx,
    worker_idx,
    enable_vehicle_routing,
):
    np.random.seed(int(args.cav_seed) + worker_idx * 100000 + episode_idx)

    obs, info = env.reset()
    env.init_infos()

    veh_obs = info["veh_obs"]
    veh_action_masks = info["action_masks"]
    next_hops = info["next_hops"]
    decision_vehicle_ids = info.get("decision_vehicle_ids", [])

    travel_time = 0
    ep_veh_reward = 0.0
    ep_reward = 0.0
    total_queued = 0.0
    pressure = 0.0
    run_step = 0
    avg_speed = 0.0
    ts_waiting_time = 0.0
    avg_density = {tls_id: 0.0 for tls_id in agent_names}
    delay_dict = {tls_id: 0.0 for tls_id in agent_names}
    pending_vehicle_experiences = {}
    signal_transitions = []
    vehicle_transitions = []
    warmup_steps = 1
    start_time = time.time()

    while env.env.agents:
        run_step += 1

        completed_vehicle_ids = []
        if enable_vehicle_routing:
            current_decision_vehicle_ids = set(decision_vehicle_ids)
            for veh_id, exp in list(pending_vehicle_experiences.items()):
                finished, is_terminal = is_vehicle_decision_finished(env, veh_id, current_decision_vehicle_ids)
                if not finished:
                    continue

                if is_terminal:
                    s_next = np.zeros_like(exp["state"])
                    mask_next = np.zeros_like(exp["mask"])
                    done_flag = True
                else:
                    s_next = veh_obs.get(veh_id, np.zeros_like(exp["state"]))
                    mask_next = veh_action_masks.get(veh_id, np.zeros_like(exp["mask"]))
                    done_flag = False

                if run_step > warmup_steps:
                    vehicle_transitions.append(make_vehicle_transition(exp, s_next, mask_next, done_flag))
                completed_vehicle_ids.append(veh_id)

            for veh_id in completed_vehicle_ids:
                pending_vehicle_experiences.pop(veh_id, None)

        signal_state = align_signal_obs(obs, agent_names, max_signal_obs_dim, add_agent_id=True)
        signal_mask = build_signal_action_masks(env, agent_names, max_signal_action_dim)
        signal_actions_array = signal_agent.choose_actions(signal_state, signal_mask, train=True)
        signal_actions = {
            tls_id: int(np.clip(signal_actions_array[idx], 0, env.action_space(tls_id).n - 1))
            for idx, tls_id in enumerate(agent_names)
        }

        route_actions = {}
        if enable_vehicle_routing:
            for veh_id in decision_vehicle_ids:
                if veh_id not in veh_obs or veh_id in pending_vehicle_experiences:
                    continue

                mask = veh_action_masks.get(veh_id)
                valid_hops = next_hops.get(veh_id, [])
                if mask is None or len(valid_hops) == 0 or np.sum(mask) <= 0:
                    continue

                action = int(vehicle_agent.choose_action((veh_obs[veh_id], mask), train=True))
                if action >= len(valid_hops):
                    continue

                route_actions[veh_id] = valid_hops[action]
                pending_vehicle_experiences[veh_id] = {
                    "state": veh_obs[veh_id].copy(),
                    "mask": mask.copy(),
                    "action": action,
                    "acc_reward": 0.0,
                    "tau": 0,
                }

        next_obs, reward, done, next_info = env.step((signal_actions, route_actions))
        next_veh_obs = next_info["veh_obs"]
        next_veh_action_masks = next_info["action_masks"]
        next_next_hops = next_info["next_hops"]
        next_decision_vehicle_ids = next_info.get("decision_vehicle_ids", [])
        veh_reward = next_info["veh_reward"]

        terminal_vehicle_ids = []
        if enable_vehicle_routing:
            for veh_id in list(pending_vehicle_experiences.keys()):
                accumulate_vehicle_reward(
                    vehicle_agent,
                    pending_vehicle_experiences[veh_id],
                    veh_reward.get(veh_id, 0.0),
                )

            for veh_id, exp in list(pending_vehicle_experiences.items()):
                finished, is_terminal = is_vehicle_decision_finished(env, veh_id, set())
                if not finished:
                    continue

                if is_terminal:
                    s_next = np.zeros_like(exp["state"])
                    mask_next = np.zeros_like(exp["mask"])
                    done_flag = True
                else:
                    s_next = next_veh_obs.get(veh_id, np.zeros_like(exp["state"]))
                    mask_next = next_veh_action_masks.get(veh_id, np.zeros_like(exp["mask"]))
                    done_flag = False

                if run_step > warmup_steps:
                    vehicle_transitions.append(make_vehicle_transition(exp, s_next, mask_next, done_flag))
                terminal_vehicle_ids.append(veh_id)

            for veh_id in terminal_vehicle_ids:
                pending_vehicle_experiences.pop(veh_id, None)

        travel_time += len(next_veh_obs)
        ep_veh_reward += np.sum(list(veh_reward.values()))
        ep_reward += np.sum(list(reward.values()))
        total_queued += np.sum(list(env.get_total_queued().values()))
        pressure += np.sum(list(env.get_pressure().values()))
        avg_speed += np.sum(list(env.get_veh_average_speed().values()))
        ts_waiting_time += np.sum(list(env.get_ts_waiting_time().values()))

        lane_density = env.get_lane_density()
        lane_density = {tls_id: np.sum(values) / len(values) for tls_id, values in lane_density.items()}
        for tls_id in avg_density:
            avg_density[tls_id] += lane_density.get(tls_id, 0.0)

        delays = env.get_delay()
        for tls_id in delay_dict:
            delay_dict[tls_id] += delays.get(tls_id, 0.0)

        if run_step > warmup_steps:
            next_signal_state = align_signal_obs(next_obs, agent_names, max_signal_obs_dim, add_agent_id=True)
            next_signal_mask = build_signal_action_masks(env, agent_names, max_signal_action_dim)
            signal_reward = np.array(
                [np.clip(reward.get(tls_id, 0.0), -100.0, 100.0) for tls_id in agent_names],
                dtype=np.float32,
            )
            signal_done = np.array([float(done.get(tls_id, False)) for tls_id in agent_names], dtype=np.float32)
            signal_transitions.append(
                (
                    signal_state,
                    signal_mask,
                    signal_actions_array,
                    signal_reward,
                    next_signal_state,
                    next_signal_mask,
                    signal_done,
                )
            )

        obs = next_obs.copy()
        veh_obs = next_veh_obs.copy()
        veh_action_masks = next_veh_action_masks.copy()
        next_hops = next_next_hops.copy()
        decision_vehicle_ids = next_decision_vehicle_ids.copy()

    if enable_vehicle_routing:
        for exp in pending_vehicle_experiences.values():
            vehicle_transitions.append(
                make_vehicle_transition(
                    exp,
                    np.zeros_like(exp["state"]),
                    np.zeros_like(exp["mask"]),
                    True,
                )
            )

    stats = {
        "travel_time": travel_time,
        "ep_veh_reward": ep_veh_reward,
        "veh_loss": 0.0,
        "ts_loss": 0.0,
        "ep_reward": ep_reward,
        "total_queued": total_queued,
        "pressure": pressure,
        "delay": env.get_mean_delay(),
        "avg_speed": avg_speed,
        "ts_waiting_time": ts_waiting_time,
        "run_step": run_step,
        "elapsed": time.time() - start_time,
    }
    return stats, avg_density, delay_dict, signal_transitions, vehicle_transitions


def worker_main(worker_idx, episode_indices, args, weight_queue, result_queue):
    try:
        net_file, route_file, _ = resolve_dataset(args)
        probe_env = Vehicle_single_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=args.num_seconds,
            fixed_ts=False,
        )
        cav_vehicle_ids, cav_penetration = build_cav_vehicle_set(probe_env, args.cav_penetration, args.cav_seed)
        enable_vehicle_routing = cav_penetration > 0.0
        probe_env.close()

        env = create_env(args, net_file, route_file, cav_vehicle_ids, enable_vehicle_routing)
        agent_names = list(env.agents)
        obs, info = env.reset()
        env.init_infos()
        signal_agent, max_signal_obs_dim, max_signal_action_dim = build_signal_agent(args, env, agent_names, obs)
        vehicle_agent = build_vehicle_agent(args, info)

        for episode_idx in episode_indices:
            weights = weight_queue.get()
            if weights is None:
                break
            load_weight_packet(signal_agent, vehicle_agent, weights)
            (
                stats,
                avg_density,
                delay_dict,
                signal_transitions,
                vehicle_transitions,
            ) = run_rollout_episode(
                args,
                env,
                agent_names,
                signal_agent,
                vehicle_agent,
                max_signal_obs_dim,
                max_signal_action_dim,
                episode_idx,
                worker_idx,
                enable_vehicle_routing,
            )
            result_queue.put(
                {
                    "type": "episode",
                    "worker_idx": worker_idx,
                    "episode_idx": episode_idx,
                    "stats": stats,
                    "avg_density": avg_density,
                    "delay_dict": delay_dict,
                    "signal_transitions": signal_transitions,
                    "vehicle_transitions": vehicle_transitions,
                }
            )

        env.close()
        result_queue.put({"type": "done", "worker_idx": worker_idx})
    except Exception as exc:
        result_queue.put({"type": "error", "worker_idx": worker_idx, "error": repr(exc)})


def learner_update(signal_agent, vehicle_agent, signal_transitions, vehicle_transitions, counters):
    for transition in signal_transitions:
        signal_agent.buffer.store(*transition)
    for transition in vehicle_transitions:
        vehicle_agent.buffer.store(*transition)

    ts_loss = 0.0
    veh_loss = 0.0

    for _ in range(len(signal_transitions)):
        if len(signal_agent.buffer) <= signal_agent.batch_size:
            break
        ts_loss = signal_agent.update()
        counters["signal"] += 1
        if counters["signal"] % 10 == 0:
            signal_agent.soft_update_target(tau=0.005)

    vehicle_update_budget = 0
    if vehicle_transitions:
        vehicle_update_budget = max(1, min(len(vehicle_transitions), max(1, len(signal_transitions) // 10)))

    for _ in range(vehicle_update_budget):
        if len(vehicle_agent.buffer) <= vehicle_agent.batch_size:
            break
        veh_loss = vehicle_agent.update_for_mutil()
        counters["vehicle"] += 1
        if counters["vehicle"] % 300 == 0:
            tau = 0.005
            for target_param, policy_param in zip(
                vehicle_agent.target_net.parameters(),
                vehicle_agent.policy_net.parameters(),
            ):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    return ts_loss, veh_loss


def split_episodes(num_episodes, workers):
    chunks = [[] for _ in range(workers)]
    for episode_idx in range(num_episodes):
        chunks[episode_idx % workers].append(episode_idx)
    return chunks


def write_summary_csv(log_dir, ordered_results):
    csv_path = log_dir / "distributed_episode_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "episode",
                "worker",
                "reward",
                "queue",
                "waiting_time",
                "vehicle_reward",
                "ts_loss",
                "veh_loss",
                "run_step",
                "elapsed",
            ]
        )
        for episode_idx in sorted(ordered_results):
            result = ordered_results[episode_idx]
            stats = result["stats"]
            writer.writerow(
                [
                    episode_idx,
                    result["worker_idx"],
                    stats["ep_reward"],
                    stats["total_queued"],
                    stats["ts_waiting_time"],
                    stats["ep_veh_reward"],
                    stats["ts_loss"],
                    stats["veh_loss"],
                    stats["run_step"],
                    stats["elapsed"],
                ]
            )


def write_episode_checkpoint(log_dir, result):
    checkpoint_dir = log_dir / "episode_metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    episode_idx = result["episode_idx"]
    worker_idx = result["worker_idx"]
    payload = {
        "episode": episode_idx,
        "worker": worker_idx,
        "stats": result["stats"],
        "avg_density": result["avg_density"],
        "delay_dict": result["delay_dict"],
    }
    path = checkpoint_dir / f"episode_{episode_idx:04d}_worker_{worker_idx}.json"
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def append_summary_row(log_dir, result):
    csv_path = log_dir / "distributed_episode_summary_live.csv"
    write_header = not csv_path.exists()
    stats = result["stats"]
    with open(csv_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        if write_header:
            writer.writerow(
                [
                    "episode",
                    "worker",
                    "reward",
                    "queue",
                    "waiting_time",
                    "vehicle_reward",
                    "ts_loss",
                    "veh_loss",
                    "run_step",
                    "elapsed",
                ]
            )
        writer.writerow(
            [
                result["episode_idx"],
                result["worker_idx"],
                stats["ep_reward"],
                stats["total_queued"],
                stats["ts_waiting_time"],
                stats["ep_veh_reward"],
                stats["ts_loss"],
                stats["veh_loss"],
                stats["run_step"],
                stats["elapsed"],
            ]
        )


def main():
    args = get_common_args()
    workers = max(1, int(getattr(args, "workers", 1)))
    net_file, route_file, num_episodes = resolve_dataset(args)
    episode_chunks = split_episodes(num_episodes, workers)

    probe_env = Vehicle_single_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=args.num_seconds,
        fixed_ts=False,
    )
    cav_vehicle_ids, cav_penetration = build_cav_vehicle_set(probe_env, args.cav_penetration, args.cav_seed)
    enable_vehicle_routing = cav_penetration > 0.0
    probe_env.set_controlled_vehicle_ids(cav_vehicle_ids if enable_vehicle_routing else set())
    agent_names = list(probe_env.agents)
    obs, info = probe_env.reset()
    probe_env.init_infos()
    signal_agent, max_signal_obs_dim, max_signal_action_dim = build_signal_agent(args, probe_env, agent_names, obs)
    del max_signal_obs_dim, max_signal_action_dim
    vehicle_agent = build_vehicle_agent(args, info)
    probe_env.close()

    route_tag = f"route_cav{int(round(cav_penetration * 100)):03d}" if enable_vehicle_routing else "no_route"
    timestamp = time.strftime("%Y%m%d-%H%M%S") + (
        f"_{args.dataset}_{args.q_target}_{route_tag}_dist{workers}_co_tsc"
    )
    log_dir = ROOT_DIR / "result" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Distributed training: dataset={args.dataset}, episodes={num_episodes}, "
        f"workers={workers}, chunks={[len(chunk) for chunk in episode_chunks]}, "
        f"q_target={args.q_target}, cav_penetration={cav_penetration:.2f}, "
        f"controlled_vehicles={len(cav_vehicle_ids)}/{len(probe_env.all_vehicle_target_road) if hasattr(probe_env, 'all_vehicle_target_road') else 'closed'}"
    )

    result_queue = mp.Queue()
    weight_queues = [mp.Queue(maxsize=1) for _ in range(workers)]
    processes = []
    initial_weights = make_weight_packet(signal_agent, vehicle_agent)
    for worker_idx, episode_indices in enumerate(episode_chunks):
        process = mp.Process(
            target=worker_main,
            args=(worker_idx, episode_indices, copy.deepcopy(args), weight_queues[worker_idx], result_queue),
            daemon=False,
        )
        process.start()
        processes.append(process)
        if episode_indices:
            weight_queues[worker_idx].put(initial_weights)

    counters = {"signal": 0, "vehicle": 0}
    completed_workers = set()
    completed_by_worker = {idx: 0 for idx in range(workers)}
    ordered_results = {}
    start_time = time.time()

    while len(completed_workers) < workers:
        try:
            message = result_queue.get(timeout=30)
        except queue.Empty:
            alive = [idx for idx, process in enumerate(processes) if process.is_alive()]
            print(f"[learner] waiting... alive_workers={alive}, elapsed={time.time() - start_time:.1f}s")
            continue

        msg_type = message.get("type")
        worker_idx = message.get("worker_idx")
        if msg_type == "error":
            raise RuntimeError(f"worker {worker_idx} failed: {message.get('error')}")

        if msg_type == "done":
            completed_workers.add(worker_idx)
            print(f"[learner] worker={worker_idx} done")
            continue

        if msg_type != "episode":
            continue

        ts_loss, veh_loss = learner_update(
            signal_agent,
            vehicle_agent,
            message["signal_transitions"],
            message["vehicle_transitions"],
            counters,
        )
        message["stats"]["ts_loss"] = ts_loss
        message["stats"]["veh_loss"] = veh_loss
        ordered_results[message["episode_idx"]] = message
        write_episode_checkpoint(log_dir, message)
        append_summary_row(log_dir, message)
        completed_by_worker[worker_idx] += 1

        print(
            f"[learner] episode={message['episode_idx']} worker={worker_idx} "
            f"reward={message['stats']['ep_reward']:.2f} queue={message['stats']['total_queued']:.1f} "
            f"ts_loss={ts_loss:.5f} veh_loss={veh_loss:.5f} "
            f"done={len(ordered_results)}/{num_episodes}"
        )

        if completed_by_worker[worker_idx] < len(episode_chunks[worker_idx]):
            weight_queues[worker_idx].put(make_weight_packet(signal_agent, vehicle_agent))

    for process in processes:
        process.join()

    for episode_idx in sorted(ordered_results):
        result = ordered_results[episode_idx]
        write_episode_logs(log_dir, result["stats"], result["avg_density"], result["delay_dict"])
    write_summary_csv(log_dir, ordered_results)

    model_dir = ROOT_DIR / "result" / "model" / f"tl_ve_{args.dataset}_{args.q_target}_{route_tag}_dist{workers}"
    signal_agent.save_model_(str(model_dir))
    if enable_vehicle_routing:
        vehicle_agent.save_model_(str(model_dir))

    print(f"Distributed training complete. log_dir={log_dir}")
    print(f"model_dir={model_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
