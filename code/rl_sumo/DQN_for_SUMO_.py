import os
import numpy as np
import sumolib
from single_agent.Q_Value.DQN import DQN
from common.arguments import get_common_args, get_DQN_args
from env.wrapper_env import wrapper_env_sumo
from until import find_next, bfs_all_paths, extract_vehicle_routes

"""
    环境封装
"""

os.environ['LIBSUMO_AS_TRACI'] = '1'

if __name__ == '__main__':
    args = get_common_args()
    args = get_DQN_args(args)
    args.env_name = 'SUMO'
    args.alg_name = 'DQN'

    net_file = 'D:/wsp_python/TSC_tiaozhanbei_compare/net_work/hangzhou_peak/hangzhou_4x4_gudang_18041610_1h.net.xml'
    # route_file = 'D:/wsp_python/sumo_veichle_traffic_light/net_work/hangzhou/Hangzhou_mixed.rou.xml'
    route_file = 'D:/wsp_python/TSC_tiaozhanbei_compare/net_work/hangzhou_peak/output.rou.xml'

    net = sumolib.net.readNet(net_file)
    all_vehicle_start_end = extract_vehicle_routes(route_file)

    all_vehicle_road = {k: bfs_all_paths(net, v[0], v[1]) for k, v in all_vehicle_start_end.items()}
    all_vehicle_target_road = {k: v[1] for k, v in all_vehicle_start_end.items()}

    ts_edge = {}
    with open('ts_edge', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                inter, roads = line.split(':')
                ts_edge[inter] = roads.split(',')

    edge_ts = {}
    for k, v in ts_edge.items():
        for v_ in v:
            edge_ts[v_] = k
    edge_list = list(edge_ts.keys())

    env = wrapper_env_sumo(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=3600, fixed_ts=False)
    obs = env.reset()

    agent_names = env.agents
    args.input_dim = env.observation_space(agent_names[0]).shape[0]
    args.output_dim = env.action_space(agent_names[0]).n

    args.add_agent_id = True

    if args.add_agent_id:
        args.input_dim = args.input_dim + len(agent_names)

    args.n_agents = len(agent_names)
    args.num_episodes = 100

    agents = DQN(args)
    running_step = 0


    for i in range(args.num_episodes):

        obs = env.reset()
        ep_reward = 0

        run_step = 0

        while env.env.agents:

            # print(env.getTSC_7Paper_obs(), 'paper_obs')
            running_step += 1
            run_step += 1
            actions = agents.choose_action_for_mutil(list(obs.values()))
            actions = dict(zip(agent_names, actions))
            # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            next_obs, reward, done, info = env.step(actions)
            ep_reward += np.sum(list(reward.values()))


            #
            agents.buffer.store(list(obs.values()), list(actions.values()), list(reward.values()),
                                list(next_obs.values()), list(done.values()))

            if (running_step % 10) == 0:
                curr_edge_id = env.get_all_vehicle_rouID()
                for k, v in curr_edge_id.items():

                    if v in edge_list:

                        phase = env.get_phase()[edge_ts[v]]

                        if v != all_vehicle_target_road[k]:

                            next_hops, rest_road = find_next(curr_edge_id[k], all_vehicle_road[k])

                            choose_hops = env.calculate_density_and_choose_next_hop(k, rest_road,
                                                                                    edge_ts[v])

                            if choose_hops is not None:
                                env.change_path(k, choose_hops)

            obs = next_obs.copy()
            agents.update_for_mutil()
            if (running_step % 100) == 0:
                agents.target_net.load_state_dict(agents.policy_net.state_dict())


    env.close()
    agents.save_model()
