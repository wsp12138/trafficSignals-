import copy
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from common.arguments import get_common_args, get_DQN_args
from single_agent.Q_Value.DQN_action_mask import action_mask_DQN
from single_agent.Q_Value.MultiAgentMaskedDQN import MultiAgentMaskedDQN
from vehicle_single_env_finally import Vehicle_single_env


os.environ["LIBSUMO_AS_TRACI"] = "1"


ROOT_DIR = Path(__file__).resolve().parent


DATASETS = {
    "nanchang_split1": {
        "net": ROOT_DIR / "net_work/nanchang_round2_split1/sumo/nanchang_round2_split1.net.xml",
        "route": ROOT_DIR / "net_work/nanchang_round2_split1/sumo/nanchang_round2_split1.rou.xml",
        "default_episodes": 100,
    },
    "hangzhou": {
        "net": ROOT_DIR / "net_work/hangzhou_peak/hangzhou_4x4_gudang_18041610_1h.net.xml",
        "route": ROOT_DIR / "net_work/hangzhou/hangzhou_4x4_gudang_18041610_1h.rou.xml",
        "default_episodes": 100,
    },
    "hangzhou_peak": {
        "net": ROOT_DIR / "net_work/hangzhou_peak/hangzhou_4x4_gudang_18041610_1h.net.xml",
        "route": ROOT_DIR / "net_work/hangzhou_peak/output.rou.xml",
        "default_episodes": 100,
    },
}


def resolve_dataset(args):
    preset = DATASETS[args.dataset]
    net_file = Path(args.net_file) if args.net_file else preset["net"]
    route_file = Path(args.route_file) if args.route_file else preset["route"]
    if not net_file.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_file}")
    if not route_file.exists():
        raise FileNotFoundError(f"SUMO route file not found: {route_file}")
    num_episodes = args.episode if args.episode != 1000 else preset["default_episodes"]
    return str(net_file), str(route_file), num_episodes


def is_vehicle_decision_finished(env, veh_id, next_decision_vehicle_ids):
    if veh_id not in env.current_veh_states:
        return True, True
    return veh_id in next_decision_vehicle_ids, False


def normalize_penetration(value):
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))


def build_cav_vehicle_set(env, penetration, seed=0):
    penetration = normalize_penetration(penetration)
    vehicle_ids = sorted(env.all_vehicle_target_road.keys())
    if penetration >= 1.0:
        return set(vehicle_ids), penetration
    if penetration <= 0.0:
        return set(), penetration

    cav_ids = set()
    for veh_id in vehicle_ids:
        digest = hashlib.md5(f"{seed}:{veh_id}".encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        if score < penetration:
            cav_ids.add(veh_id)
    return cav_ids, penetration


def build_signal_action_masks(env, agent_names, max_action_dim, enforce_min_green=True):
    masks = np.zeros((len(agent_names), max_action_dim), dtype=bool)
    for idx, tls_id in enumerate(agent_names):
        action_count = env.action_space(tls_id).n
        masks[idx, :action_count] = True

        if not enforce_min_green:
            continue

        try:
            ts = env.env.sumo_env.traffic_signals[tls_id]
            min_hold = ts.min_green + ts.yellow_time
            if ts.time_since_last_phase_change < min_hold:
                current_phase = int(ts.green_phase)
                masks[idx, :] = False
                if 0 <= current_phase < action_count:
                    masks[idx, current_phase] = True
                else:
                    masks[idx, 0] = True
        except Exception:
            pass

        if not masks[idx].any():
            masks[idx, 0] = True
    return masks


def align_signal_obs(obs, agent_names, max_obs_dim, add_agent_id=True, add_structure_mask=True):
    id_dim = len(agent_names) if add_agent_id else 0
    mask_dim = max_obs_dim if add_structure_mask else 0
    aligned = np.zeros((len(agent_names), max_obs_dim + mask_dim + id_dim), dtype=np.float32)
    for idx, tls_id in enumerate(agent_names):
        raw_obs = obs.get(tls_id)
        if raw_obs is None:
            raw = np.zeros(0, dtype=np.float32)
        else:
            raw = np.asarray(raw_obs, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
        use_len = min(len(raw), max_obs_dim)
        aligned[idx, :use_len] = raw[:use_len]
        if add_structure_mask:
            aligned[idx, max_obs_dim : max_obs_dim + use_len] = 1.0
        if add_agent_id:
            aligned[idx, max_obs_dim + mask_dim + idx] = 1.0
    return aligned


def build_signal_agent(args, env, agent_names, obs):
    signal_args = get_DQN_args(copy.copy(args))
    max_obs_dim = max(len(np.asarray(obs[tls_id])) for tls_id in agent_names)
    max_action_dim = max(env.action_space(tls_id).n for tls_id in agent_names)

    signal_args.env_name = "SUMO"
    signal_args.alg_name = f"co_tsc_{args.dataset}"
    signal_args.n_agents = len(agent_names)
    signal_args.add_agent_id = True
    signal_args.add_structure_mask = True
    signal_args.input_dim = max_obs_dim * 2 + len(agent_names)
    signal_args.max_output_dim = max_action_dim

    # Nanchang is heterogeneous and much noisier than Hangzhou. The masked
    # shared signal policy needs a larger buffer and slower exploration decay.
    signal_args.hidden_dim = 128
    signal_args.gamma = 0.95
    signal_args.capacity = 50000
    signal_args.batch_size = 32
    signal_args.lr = 0.0003
    signal_args.epsilon_start = 1.0
    signal_args.epsilon_end = 0.05
    signal_args.epsilon_decay = 10000
    signal_args.use_double_dqn = args.q_target == "ddqn"

    agent = MultiAgentMaskedDQN(signal_args)
    return agent, max_obs_dim, max_action_dim


def build_vehicle_agent(args, initial_info):
    vehicle_args = get_DQN_args(copy.copy(args))
    veh_obs = initial_info.get("veh_obs", {})
    sample_veh_id = next(iter(veh_obs.keys()), None)

    vehicle_args.env_name = "SUMO"
    vehicle_args.alg_name = f"vehicle_route_plan_{args.dataset}"
    vehicle_args.input_dim = len(veh_obs[sample_veh_id]) if sample_veh_id else 12
    vehicle_args.max_output_dim = 3
    vehicle_args.hidden_dim = 64
    vehicle_args.gamma = 0.99
    vehicle_args.tau_discount_clip = 10.0
    vehicle_args.capacity = 50000
    vehicle_args.epsilon_start = 0.95
    vehicle_args.epsilon_end = 0.05
    vehicle_args.epsilon_decay = 10000
    vehicle_args.lr = 0.0003
    vehicle_args.n_agents = 1
    vehicle_args.use_double_dqn = args.q_target == "ddqn"
    return action_mask_DQN(vehicle_args)


def accumulate_vehicle_reward(agent, exp, step_reward):
    tau_clip = float(getattr(agent.args, "tau_discount_clip", 10.0))
    reward_tau = min(float(exp["tau"]), tau_clip)
    exp["acc_reward"] += (agent.gamma ** reward_tau) * float(step_reward)
    exp["tau"] += 1


def store_vehicle_transition(agent, exp, next_state, next_mask, done_flag):
    clipped_reward = np.clip(exp["acc_reward"], -50.0, 250.0)
    agent.buffer.store(
        (exp["state"], exp["mask"]),
        exp["action"],
        clipped_reward,
        (next_state, next_mask),
        done_flag,
        max(exp["tau"], 1),
    )


def write_episode_logs(log_dir, stats, avg_density, delay_dict):
    file_map = {
        "route_plan_dqn_peak_travel_time_1.txt": stats["travel_time"],
        "route_plan_dqn_peak_ep_veh_reward_1.txt": stats["ep_veh_reward"],
        "route_plan_dqn_peak_veh_loss_1.txt": stats["veh_loss"],
        "route_plan_dqn_peak_ts_loss_1.txt": stats["ts_loss"],
        "reward_1.txt": stats["ep_reward"],
        "queue_1.txt": stats["total_queued"],
        "route_plan_dqn_peak_pressure_1.txt": stats["pressure"],
        "route_plan_dqn_peak_delay_1.txt": stats["delay"],
        "route_plan_dqn_peak_avg_speed_1.txt": stats["avg_speed"],
        "route_plan_dqn_peak_ts_waiting_time_1.txt": stats["ts_waiting_time"],
    }
    for filename, value in file_map.items():
        with open(log_dir / filename, "a", encoding="utf-8") as file_obj:
            file_obj.write(str(value) + "\n")

    with open(log_dir / "route_plan_dqn_peak_ts_density_1.txt", "a", encoding="utf-8") as file_obj:
        for tls_id, value in avg_density.items():
            file_obj.write(f"{tls_id}:{value}\n")
    with open(log_dir / "route_plan_dqn_peak_ts_delays_1.txt", "a", encoding="utf-8") as file_obj:
        for tls_id, value in delay_dict.items():
            file_obj.write(f"{tls_id}:{value}\n")


def main():
    args = get_common_args()
    net_file, route_file, num_episodes = resolve_dataset(args)

    env = Vehicle_single_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.use_gui,
        num_seconds=args.num_seconds,
        fixed_ts=False,
    )
    cav_vehicle_ids, cav_penetration = build_cav_vehicle_set(env, args.cav_penetration, args.cav_seed)
    enable_vehicle_routing = cav_penetration > 0.0
    env.set_controlled_vehicle_ids(cav_vehicle_ids if enable_vehicle_routing else set())

    agent_names = list(env.agents)
    obs, info = env.reset()
    env.init_infos()

    signal_agent, max_signal_obs_dim, max_signal_action_dim = build_signal_agent(args, env, agent_names, obs)
    vehicle_agent = build_vehicle_agent(args, info)

    route_tag = f"route_cav{int(round(cav_penetration * 100)):03d}" if enable_vehicle_routing else "no_route"
    timestamp = time.strftime("%Y%m%d-%H%M%S") + f"_{args.dataset}_{args.q_target}_{route_tag}_co_tsc"
    log_dir = ROOT_DIR / "result" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Dataset={args.dataset}, agents={len(agent_names)}, "
        f"max_signal_obs={max_signal_obs_dim}, max_signal_actions={max_signal_action_dim}, "
        f"q_target={args.q_target}, vehicle_routing={enable_vehicle_routing}, "
        f"cav_penetration={cav_penetration:.2f}, "
        f"controlled_vehicles={len(cav_vehicle_ids)}/{len(env.all_vehicle_target_road)}, "
        f"episodes={num_episodes}"
    )

    running_step = 0
    signal_update_count = 0
    vehicle_update_count = 0
    vehicle_target_update_freq = 300
    warmup_steps = 1

    for episode_idx in range(num_episodes):
        start_time = time.time()

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
        ts_loss = 0.0
        veh_loss = 0.0
        pending_vehicle_experiences = {}

        while env.env.agents:
            running_step += 1
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
                        store_vehicle_transition(vehicle_agent, exp, s_next, mask_next, done_flag)
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
                    if veh_id not in cav_vehicle_ids:
                        continue

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
                        store_vehicle_transition(vehicle_agent, exp, s_next, mask_next, done_flag)
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
                signal_agent.buffer.store(
                    signal_state,
                    signal_mask,
                    signal_actions_array,
                    signal_reward,
                    next_signal_state,
                    next_signal_mask,
                    signal_done,
                )

            obs = next_obs.copy()
            veh_obs = next_veh_obs.copy()
            veh_action_masks = next_veh_action_masks.copy()
            next_hops = next_next_hops.copy()
            decision_vehicle_ids = next_decision_vehicle_ids.copy()

            if len(signal_agent.buffer) > signal_agent.batch_size:
                ts_loss = signal_agent.update()
                signal_update_count += 1
                if signal_update_count % 10 == 0:
                    signal_agent.soft_update_target(tau=0.005)

            if enable_vehicle_routing and len(vehicle_agent.buffer) > vehicle_agent.batch_size:
                if completed_vehicle_ids or terminal_vehicle_ids or run_step % 10 == 0:
                    veh_loss = vehicle_agent.update_for_mutil()
                    vehicle_update_count += 1
                    if vehicle_update_count % vehicle_target_update_freq == 0:
                        tau = 0.005
                        for target_param, policy_param in zip(
                            vehicle_agent.target_net.parameters(),
                            vehicle_agent.policy_net.parameters(),
                        ):
                            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

            if running_step % 100 == 0:
                print(
                    f"episode={episode_idx} step={run_step} "
                    f"ts_loss={ts_loss:.5f} veh_loss={veh_loss:.5f} "
                    f"queue={total_queued:.1f} wait={ts_waiting_time:.1f} "
                    f"eps_tl={signal_agent.epsilon(signal_agent.frame):.3f} "
                    f"eps_veh={vehicle_agent.epsilon(vehicle_agent.frame):.3f} "
                    f"elapsed={time.time() - start_time:.1f}s"
                )

        if enable_vehicle_routing:
            for exp in pending_vehicle_experiences.values():
                store_vehicle_transition(
                    vehicle_agent,
                    exp,
                    np.zeros_like(exp["state"]),
                    np.zeros_like(exp["mask"]),
                    True,
                )

        stats = {
            "travel_time": travel_time,
            "ep_veh_reward": ep_veh_reward,
            "veh_loss": veh_loss,
            "ts_loss": ts_loss,
            "ep_reward": ep_reward,
            "total_queued": total_queued,
            "pressure": pressure,
            "delay": env.get_mean_delay(),
            "avg_speed": avg_speed,
            "ts_waiting_time": ts_waiting_time,
        }
        write_episode_logs(log_dir, stats, avg_density, delay_dict)

        print(
            f"Episode {episode_idx}: ep_reward={ep_reward:.2f}, veh_loss={veh_loss:.6f}, "
            f"ts_loss={ts_loss:.6f}, waiting_time={ts_waiting_time:.0f}, "
            f"travel_time={travel_time}, eps_tl={signal_agent.epsilon(signal_agent.frame):.3f}, "
            f"eps_veh={vehicle_agent.epsilon(vehicle_agent.frame):.3f}"
        )

    env.close()

    model_dir = ROOT_DIR / "result" / "model" / f"tl_ve_{args.dataset}_{args.q_target}_{route_tag}"
    signal_agent.save_model_(str(model_dir))
    if enable_vehicle_routing:
        vehicle_agent.save_model_(str(model_dir))


if __name__ == "__main__":
    main()
