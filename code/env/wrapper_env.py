from collections import defaultdict
from itertools import chain

import numpy as np
from sumo_rl.environment.env import SumoEnvironmentPZ
from pettingzoo.utils.conversions import parallel_wrapper_fn, aec_to_parallel
from pettingzoo.utils import agent_selector, wrappers
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar
import gymnasium

from common.utils import DefaultObservationFunction

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    aec_env = wrappers.AssertOutOfBoundsWrapper(env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)

    parallel_env = aec_to_parallel(aec_env)
    parallel_env.raw_env = env
    parallel_env.sumo_env = env.env
    return parallel_env


observation_spaces: Dict[
    AgentID, gymnasium.spaces.Space
]


class wrapper_env_sumo:
    def __init__(self, **kwargs):
        # self.env = sumo_rl.parallel_env(**kwargs)
        self.env = env(**kwargs)
        self.env.reset()
        self.agents = self.env.agents
        self.ts = self.env.sumo_env.traffic_signals
        self.observation_spaces = {t: DefaultObservationFunction(self.ts[t]).observation_space() for t in self.ts}

        self.last_measure = {t: 0 for t in self.ts}

    def observation_space(self, agent):

        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self):

        # print(self.ts['intersection_1_1'].get_accumulated_waiting_time_per_lane(), 'ts')
        obs = self.env.reset()
        self.last_measure = {t: 0 for t in self.ts}
        # obs = self.get_obs(obs)
        # obs = self.getTSC_7Paper_obs()
        return obs

    def step(self, actions):
        next_obs, reward, done, info, _ = self.env.step(actions)
        # next_obs = self.get_obs(next_obs)
        # next_obs = self.getTSC_7Paper_obs()
        # reward = self.getTSC_7Paper_reward()
        # person_reward = self._diff_waiting_time_reward()
        # total_reward = {k: reward[k] + person_reward[k] for k in reward}
        return next_obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        return self.env.close()

    # def get_TSC_7_info(self):
    #     lane_wait_time = {t: self.env.sumo_env.traffic_signals[t].get_accumulated_waiting_time_per_lane()[:12] for t in
    #                       self.ts}
    #     lane_queue_length = {t: [self.env.sumo_env.sumo.lane.getLastStepHaltingNumber(lane) for lane in
    #                              self.env.sumo_env.traffic_signals[t].lanes][:12] for t in self.ts}
    #     wait_person_time, wait_person_number = self.get_person_wait_time_and_number_()
    #     return lane_wait_time, lane_queue_length, wait_person_time, wait_person_number
    #
    # def getTSC_7Paper_obs(self):
    #     lane_wait_time, lane_queue_length, wait_person_time, wait_person_number = self.get_TSC_7_info()
    #     obs = {t: lane_wait_time[t] + lane_queue_length[t] + [wait_person_time[t]] + [wait_person_number[t]] for t in
    #            self.ts}
    #     return obs
    #
    # def getTSC_7Paper_reward(self):
    #     lane_wait_time, lane_queue_length, wait_person_time, wait_person_number = self.get_TSC_7_info()
    #     lane_queue_length = {k: np.array(v) * 1.5 for k, v in lane_queue_length.items()}
    #     wait_person_time = {k: np.array(v) * 2 for k, v in wait_person_time.items()}
    #     wait_person_number = {k: np.array(v) * 2 for k, v in wait_person_number.items()}
    #
    #     reward = {
    #         k: -(np.sum(lane_wait_time[k]) + np.sum(lane_queue_length[k]) + wait_person_time[k] + wait_person_number[k])
    #         for k in
    #         self.ts}
    #     return reward

    def _diff_waiting_time_reward(self):
        person_wait_time, _ = self.get_person_wait_time_and_number_()
        person_reward = {k: self.last_measure[k] - person_wait_time[k] for k in self.last_measure.keys()}
        self.last_measure = person_wait_time.copy()
        return person_reward

    def get_phase(self):
        phase_id = {ts_id: [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)] for ts_id, ts in
                    self.env.sumo_env.traffic_signals.items()}
        return phase_id

    def get_all_lanes(self):
        lanse_and_out_lanse = {
            ts: [self.env.env.traffic_signals[ts].get_lanes(), self.env.env.traffic_signals[ts].get_out_lanes()] for ts
            in
            self.env.env.traffic_signals}
        return lanse_and_out_lanse

    def get_vehicle_number(self):
        lanes_density = {ts: self.env.env.traffic_signals[ts].get_lanes_density() for ts in
                         self.env.env.traffic_signals}
        out_lanes_density = {ts: self.env.env.traffic_signals[ts].get_out_lanes_density() for ts in
                             self.env.env.traffic_signals}
        return lanes_density, out_lanes_density

    def get_lanes_for_dict(self):
        return {ts: self.env.env.traffic_signals[ts].get_lanes_density_for_dict() for ts in
                self.env.env.traffic_signals}

    def get_out_lanes_for_dict(self):
        return {ts: self.env.env.traffic_signals[ts].get_out_lanes_density_for_dict() for ts in
                self.env.env.traffic_signals}

    def get_lanes_number(self):
        lanes = {ts: len(self.env.env.traffic_signals[ts].get_lanes_number()) for ts in self.env.env.traffic_signals}
        return lanes

    def get_phase_id_number(self):
        phase_id_num = {ts_id: ts.num_green_phases for ts_id, ts in self.env.sumo_env.traffic_signals.items()}
        return phase_id_num

    def get_total_queued(self):
        return self.env.get_total_queued()
        # total_queued = {ts: self.env.sumo_env.traffic_signals[ts].get_total_queued() for ts in
        #                 self.env.sumo_env.traffic_signals}
        # return total_queued

    def get_pressure(self):
        return self.env.get_pressure()

        # pressure = {ts: self.env.sumo_env.traffic_signals[ts].get_pressure() for ts in
        #             self.env.sumo_env.traffic_signals}
        # return pressure

    def get_all_edge(self):
        all_edge = self.env.sumo_env.sumo.edge.getIDList()
        return all_edge

    def get_person_lane(self):

        person_lane = defaultdict(list)
        lanse = {ts: self.env.env.traffic_signals[ts].get_lanes() for ts in self.env.env.traffic_signals}
        # print(lanse,'lanse')
        for key in lanse.keys():
            for lane in lanse[key]:
                # print(lane,self.env.sumo.lane.getAllowed(lane))
                if 'pedestrian' in self.env.env.sumo.lane.getAllowed(lane):
                    person_lane[key].append(lane)

        return person_lane

    # def get_person_number(self):
    #
    #     edge_person = {t: [] for t in self.ts.keys()}
    #
    #     person_ids = self.env.sumo_env.sumo.person.getIDList()
    #
    #     for person_id in person_ids:
    #         edge_id = self.env.sumo_env.sumo.person.getRoadID(person_id)
    #         print(edge_id)
    #         lane_id = self.env.sumo_env.sumo.person.getLaneID(person_id)
    #         ts_id = edge_id.split('_', 1)[1][:3]
    #         ts_ = 'intersection_' + ts_id
    #
    #         edge_person[ts_].append(person_id)
    #     return edge_person
    #
    # def get_person_wait_time_and_number(self, ):
    #     halting_person_wait_time = {t: [] for t in self.ts.keys()}
    #     halting_person_number = {t: [] for t in self.ts.keys()}
    #     edge_person = self.get_person_number()
    #     print(edge_person, 'edge_person')
    #     for key in edge_person.keys():
    #         person_ids = edge_person[key]
    #         for person_id in person_ids:
    #             wait_time = self.env.sumo_env.sumo.person.getWaitingTime(person_id)
    #             # speed = self.env.sumo_env.sumo.person.getSpeed(person_id)
    #             if wait_time > 0:
    #                 halting_person_wait_time[key].append(wait_time)
    #                 halting_person_number[key].append(1)
    #
    #     halting_person_wait_time = {k: sum(v) for k, v in halting_person_wait_time.items()}
    #     halting_person_number = {k: sum(v) for k, v in halting_person_number.items()}
    #
    #     return halting_person_wait_time, halting_person_number

    # def get_person_wait_time_and_number_(self):
    #     person_ids = self.env.sumo_env.sumo.person.getIDList()
    #
    #     halting_person_wait_time = {t: [] for t in self.ts.keys()}
    #     halting_person_number = {t: [] for t in self.ts.keys()}
    #
    #     for person_id in person_ids:
    #         edge_id = self.env.sumo_env.sumo.person.getRoadID(person_id)
    #
    #         ts_id = edge_id.split('_', 1)[1][:3]
    #
    #         ts_ = 'intersection_' + ts_id
    #
    #         wait_time = self.env.sumo_env.sumo.person.getWaitingTime(person_id)
    #         # speed = self.env.sumo_env.sumo.person.getSpeed(person_id)
    #         if wait_time > 0:
    #             halting_person_wait_time[ts_].append(wait_time)
    #             halting_person_number[ts_].append(1)
    #
    #     halting_person_wait_time = {k: sum(v) for k, v in halting_person_wait_time.items()}
    #     halting_person_number = {k: sum(v) for k, v in halting_person_number.items()}
    #
    #     return halting_person_wait_time, halting_person_number

    # def get_obs(self, obs):
    #     wait_person_time, wait_person_number = self.get_person_wait_time_and_number_()
    #     for key in obs.keys():
    #         obs[key] = np.concatenate(
    #             [obs[key], np.array([wait_person_number[key], wait_person_time[key]], dtype=obs[key].dtype)]
    #         )
    #
    #         # obs[key] = np.array(list(obs[key]) + [wait_person_number[key], wait_person_time[key]], dtype=np.float32)
    #     return obs

    def get_ambulance_id(self):
        vehicle_ids = self.env.sumo_env.sumo.vehicle.getIDList()
        for id in vehicle_ids:
            if self.env.sumo_env.sumo.vehicle.getTypeID(id) == 'ambulance':
                return id
        return None

    # 每次道路中最多出现一个救护车
    def get_vehicle_rouID(self):
        id = self.get_ambulance_id()
        if id is not None:
            curr_edge = self.env.sumo_env.sumo.vehicle.getRoadID(id)
            return curr_edge
        return None

    def change_ambulance(self, next_hops):

        vehicle_ids = self.env.sumo_env.sumo.vehicle.getIDList()
        for id in vehicle_ids:
            if self.env.sumo_env.sumo.vehicle.getTypeID(id) == 'ambulance':
                # target_edge = self.env.sumo_env.sumo.vehicle.getRoute(id)[-1]

                self.env.sumo_env.sumo.vehicle.changeTarget(id, next_hops)

    # def calculate_density_and_choose_next_hop(self, edge_id):
    #     edge_density = []
    #     for id in edge_id:
    #         print(id, 'id')
    #         lane_count = self.env.sumo_env.sumo.edge.getLaneNumber(id)-1
    #         print(lane_count, 'lane_count')
    #         lanes = [f"{id}_{i + 1}" for i in range(lane_count)]
    #         print(lanes, 'lanes')
    #         lanes_length = {lane: self.env.sumo_env.sumo.lane.getLength(lane) for lane in lanes}
    #         lanes_density = [
    #             self.env.sumo_env.sumo.lane.getLastStepVehicleNumber(lane)
    #             / (lanes_length[lane] / (2.5 + self.env.sumo_env.sumo.lane.getLastStepLength(lane)))
    #             for lane in lanes
    #         ]
    #         edge_density.append(np.sum(lanes_density))
    #
    #     print(edge_density, 'edge_density')

    def calculate_density_and_choose_next_hop(self, veh_id, rest_road, ts_id):
        path_length = [self.calucate_length(p) for p in rest_road]
        if len(path_length) > 1:
            max_length = np.max(path_length)
            path_length = [p / max_length for p in path_length]
        path_density = []

        first_hop = [road[0] for road in rest_road]

        gamma = 0.9
        edge_id = set(chain(*rest_road))

        edge_density = {}

        for id in edge_id:
            lane_count = self.env.sumo_env.sumo.edge.getLaneNumber(id) - 1
            lanes = [f"{id}_{i + 1}" for i in range(lane_count)]
            lanes_length = {lane: self.env.sumo_env.sumo.lane.getLength(lane) for lane in lanes}
            lanes_density = [
                self.env.sumo_env.sumo.lane.getLastStepVehicleNumber(lane)
                / (lanes_length[lane] / (2.5 + self.env.sumo_env.sumo.lane.getLastStepLength(lane)))
                for lane in lanes
            ]
            edge_density[id] = np.sum(lanes_density)

        rest_road_density = [[edge_density[edge] for edge in road] for road in rest_road]

        # rest_road_value = {k: 1 for k in next_hop}
        for i in range(len(rest_road_density)):
            # k = rest_road[i][0]

            gamma_sum = 0
            for index, value in enumerate(rest_road_density[i]):
                gamma_sum += value * pow(gamma, index)
            path_density.append(gamma_sum)
            # rest_road_value[k] = min(rest_road_value[k], gamma_sum)

        # print(rest_road_value, 'rest_road_value')

        in_lanes, out_lanes = self.get_allowed_movements(ts_id)
        current_lane = self.env.sumo_env.sumo.vehicle.getLaneID(veh_id)
        # print(current_lane, 'current_lane')
        # print(in_lanes, out_lanes, 'in_lanes, out_lanes')
        # print(next_hop, 'next_hop')

        # 根据剩余时间来算相位的具体值 然后占0.2
        allow_pass = [0] * len(first_hop)

        out_edge = set()
        if current_lane in in_lanes:
            for lane in out_lanes:
                out_edge.add(lane.rsplit('_', 1)[0])
        if len(out_edge) > 0:
            for i in range(len(first_hop)):
                if first_hop[i] in out_edge:
                    allow_pass[i] = -0.01

        res = [allow_pass[i] + 0.4 * path_length[i] + 1.2 * path_density[i] for i in range(len(allow_pass))]

        if len(res) < 1:
            return None
        return first_hop[res.index(np.min(res))]

        # print(current_edge, 'current_edge')
        # print(con_Links, 'con_Links')
        # print(rest_road_value, 'rest_road_value')
        # print('--------------------------------')

        # if len(rest_road_value.values()) == 0:
        #     return None
        #
        # min_value = min(rest_road_value.values())
        # return [key for key, value in rest_road_value.items() if value == min_value][0]

    def get_vehicle_travel_time(self):
        id = self.get_ambulance_id()
        if id is not None:
            return self.env.sumo_env.sumo.vehicle.getAccumulatedWaitingTime(id)

        return None

    def get_all_vehicle_id(self):
        vehicle_ids = self.env.sumo_env.sumo.vehicle.getIDList()

        return vehicle_ids

    def get_all_vehicle_rouID(self):
        ids = self.get_all_vehicle_id()
        curr_edge = {}
        if len(ids) > 0:
            for id in ids:
                curr_edge[id] = self.env.sumo_env.sumo.vehicle.getRoadID(id)

        return curr_edge

    def change_path(self, id, next_hops):
        self.env.sumo_env.sumo.vehicle.changeTarget(id, next_hops)

    def get_all_obs(self):
        return self.env.sumo_env._compute_observations()

    # def getControlledLinks(self, ts_id):
    #     edge_ts = defaultdict(set)
    #     con_links = self.env.sumo_env.sumo.trafficlight.getControlledLinks(ts_id)
    #
    #     for link in con_links:
    #         with_end = int(link[0][0].rsplit('_', 1)[1])
    #         if with_end != 0:
    #             k = link[0][0].rsplit('_', 1)[0]
    #             v = link[0][1].rsplit('_', 1)[0]
    #
    #             edge_ts[k].add(v)
    #     return edge_ts

    def get_allowed_movements(self, tls_id):

        links = self.env.sumo_env.sumo.trafficlight.getControlledLinks(tls_id)  # 每个 phase 对应的 in/out/via lane
        state = self.env.sumo_env.sumo.trafficlight.getRedYellowGreenState(tls_id)

        in_lanes = []
        out_lanes = []
        for i, link_group in enumerate(links):
            if state[i] == 'G':  # 当前 link 被绿灯控制
                for in_lane, out_lane, _ in link_group:
                    in_lanes.append(in_lane)
                    out_lanes.append(out_lane)
        # remaining_time = self.env.sumo_env.sumo.trafficlight.getNextSwitch(
        #     tls_id) - self.env.sumo_env.sumo.simulation.getTime()
        return in_lanes, out_lanes

    def calucate_length(self, path):

        sum_length = 0
        for p in path:
            # lane_ids = self.env.sumo_env.sumo.edge.getLaneIDs(p)
            lane_length = self.env.sumo_env.sumo.lane.getLength(p + '_0')
            sum_length += lane_length
        return sum_length

    def get_vehicle_distance_tsc(self, vehicle_id):
        tls_info = self.env.sumo_env.sumo.vehicle.getNextTLS(vehicle_id)

        if len(tls_info) > 0:
            return tls_info[0][2]
