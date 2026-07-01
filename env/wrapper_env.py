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

    def init_infos(self):
        self.enter_lane = self.get_lanes_for_dict()
        # self.vehicle_prev_lane = {}
        # self.enter_lane_time = {}
        self.enter_lane_list = sum(list(self.enter_lane.values()), [])
        # self.vehicle_travel_time_store = {}
        self.vehicle_distance_store = {}
        self.lane_max_speed = {k: self.env.sumo_env.sumo.lane.getMaxSpeed(k) for k in self.enter_lane_list}
        # self.last_travel_moment = {}
        self.prev_vehicle_distance = {}
        self.travel_time = {}
        self.flag_accident = 0
        self.flag_control = 0
        self.weather_flag = {}
        self.blocked_edges = set()
        self.veh_free_flow_times = {}
        self.veh_completed_free_flow_times = {}
        self.veh_last_physical_edge = {}
        self.veh_origin_tls = {}
        self.veh_delay_times = []
        self.veh_delay_sums_by_tls = {tls_id: 0.0 for tls_id in self.enter_lane}
        self.veh_delay_counts_by_tls = {tls_id: 0 for tls_id in self.enter_lane}
        self._pending_trip_delay_by_tls = {tls_id: 0.0 for tls_id in self.enter_lane}
        self._last_trip_delay_update_time = None
        # ====== 【新增】：用于统计旅行时间的变量 ======
        self.veh_start_times = {}  # 记录每辆车的出发时间
        self.veh_travel_times = []  # 存储所有已到达车辆的旅行时间

    def observation_space(self, agent):

        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self):

        # print(self.ts['intersection_1_1'].get_accumulated_waiting_time_per_lane(), 'ts')
        obs = self.env.reset()
        self.last_measure = {t: 0 for t in self.ts}
        self.flag_accident = 0
        self.flag_control = 0

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

    def cal_info(self):
        pass

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

    # def get_average(self):
    #     return self.env.

    def get_vehicle_number(self):
        lanes_density = {ts: self.env.env.traffic_signals[ts].get_lanes_density() for ts in
                         self.env.env.traffic_signals}
        out_lanes_density = {ts: self.env.env.traffic_signals[ts].get_out_lanes_density() for ts in
                             self.env.env.traffic_signals}
        return lanes_density, out_lanes_density

    def get_lanes_for_dict(self):
        return {ts: self.env.sumo_env.traffic_signals[ts].get_lanes() for ts in
                self.env.sumo_env.traffic_signals}
        # return {ts: self.env.env.traffic_signals[ts].get_lanes_density_for_dict() for ts in
        #         self.env.env.traffic_signals}

    def get_out_lanes_for_dict(self):
        """
        获取所有路口出口车道的密度字典。
        【修复 sumo-rl 底层 get_out_lanes_density_for_dict 遍历 self.lanes 导致的 KeyError Bug】
        """
        out_lanes_density_dict = {}
        for ts_id, ts in self.env.sumo_env.traffic_signals.items():
            # 1. 获取该路口所有的出口车道 ID 列表
            out_lanes = ts.get_out_lanes()

            # 2. 获取对应的出口车道密度列表
            densities = ts.get_out_lanes_density()

            # 3. 手动拉链(zip)拼接成字典，规避底层 Bug
            if isinstance(densities, dict):
                # 如果未来 sumo-rl 修复了并且返回字典，直接兼容
                out_lanes_density_dict[ts_id] = densities
            else:
                # 正常情况 densities 返回的是一个列表，我们用 zip 把它和 out_lanes 一一对应起来
                out_lanes_density_dict[ts_id] = {lane: min(1.0, density) for lane, density in zip(out_lanes, densities)}

        return out_lanes_density_dict

    # def get_out_lanes_for_dict(self):
    #     # print(self.env.sumo_env.traffic_signals, 'env-------------------------------')
    #     # return {ts: self.env.traffic_signals[ts].get_out_lanes_density_for_dict() for ts in
    #     #         self.env.traffic_signals}
    #     return {ts: self.env.sumo_env.traffic_signals[ts].get_out_lanes_density_for_dict() for ts in
    #             self.env.sumo_env.traffic_signals}

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

    def get_veh_average_speed(self):
        return self.env.get_veh_average_speed()

    def get_travel_time(self):
        return self.env.get_travel_time()

    # 整个路网的平均等待时间
    def get_ts_waiting_time(self):
        return self.env.get_ts_wait_time()
        # pressure = {ts: self.env.sumo_env.traffic_signals[ts].get_pressure() for ts in
        #             self.env.sumo_env.traffic_signals}
        # return pressure

    def get_lane_density(self):
        return self.env.get_lane_density()

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

    # def calculate_density_and_choose_next_hop(self, veh_id, rest_road, ts_id):
    #     path_length = [self.calucate_length(p) for p in rest_road]
    #     if len(path_length) > 1:
    #         max_length = np.max(path_length)
    #         path_length = [p / max_length for p in path_length]
    #     path_density = []
    #
    #     first_hop = [road[0] for road in rest_road]
    #
    #     gamma = 0.9
    #     edge_id = set(chain(*rest_road))
    #
    #     edge_density = {}
    #     tau_factor = getattr(self, 'current_tau_factor', 1.0)
    #
    #     for id in edge_id:
    #         lane_count = self.env.sumo_env.sumo.edge.getLaneNumber(id) - 1
    #         lanes = [f"{id}_{i + 1}" for i in range(lane_count)]
    #         lanes_length = {lane: self.env.sumo_env.sumo.lane.getLength(lane) for lane in lanes}
    #         lanes_density = [
    #             self.env.sumo_env.sumo.lane.getLastStepVehicleNumber(lane)
    #             / (lanes_length[lane] / (2.5 + self.env.sumo_env.sumo.lane.getLastStepLength(lane)))
    #             for lane in lanes
    #         ]
    #         edge_density[id] = np.sum(lanes_density)
    #
    #     rest_road_density = [[edge_density[edge] for edge in road] for road in rest_road]
    #
    #     # rest_road_value = {k: 1 for k in next_hop}
    #     for i in range(len(rest_road_density)):
    #         # k = rest_road[i][0]
    #
    #         gamma_sum = 0
    #         for index, value in enumerate(rest_road_density[i]):
    #             gamma_sum += value * pow(gamma, index)
    #         path_density.append(gamma_sum)
    #         # rest_road_value[k] = min(rest_road_value[k], gamma_sum)
    #
    #     # print(rest_road_value, 'rest_road_value')
    #
    #     in_lanes, out_lanes = self.get_allowed_movements(ts_id)
    #     current_lane = self.env.sumo_env.sumo.vehicle.getLaneID(veh_id)
    #     # print(current_lane, 'current_lane')
    #     # print(in_lanes, out_lanes, 'in_lanes, out_lanes')
    #     # print(next_hop, 'next_hop')
    #
    #     # 根据剩余时间来算相位的具体值 然后占0.2
    #     allow_pass = [0] * len(first_hop)
    #
    #     out_edge = set()
    #     if current_lane in in_lanes:
    #         for lane in out_lanes:
    #             out_edge.add(lane.rsplit('_', 1)[0])
    #     if len(out_edge) > 0:
    #         for i in range(len(first_hop)):
    #             if first_hop[i] in out_edge:
    #                 allow_pass[i] = -0.01
    #
    #     res = [allow_pass[i] + 0.4 * path_length[i] + 1.2 * path_density[i] for i in range(len(allow_pass))]
    #
    #     if len(res) < 1:
    #         return None
    #     return first_hop[res.index(np.min(res))]

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

        # ---------------------------------------------------------
        # [修改点]：获取当前天气的 tau_factor，若未设置则默认为晴天 1.0
        # ---------------------------------------------------------
        tau_factor = getattr(self, 'current_tau_factor', 1.0)

        for id in edge_id:
            lane_count = self.env.sumo_env.sumo.edge.getLaneNumber(id) - 1
            lanes = [f"{id}_{i + 1}" for i in range(lane_count)]

            lanes_density_sum = 0
            for lane in lanes:
                L = self.env.sumo_env.sumo.lane.getLength(lane)
                n = self.env.sumo_env.sumo.lane.getLastStepVehicleNumber(lane)

                # 获取车道上车辆的平均长度，如果车道为空，SUMO可能会返回 0.0，此时给个默认车长 5.0
                l_avg = self.env.sumo_env.sumo.lane.getLastStepLength(lane)
                if l_avg == 0.0:
                    l_avg = 5.0

                # 计算静态物理最大容量
                static_capacity = L / (2.5 + l_avg)
                if static_capacity <= 0:
                    static_capacity = 1.0

                # 【核心逻辑】：用天气因子动态缩减道路的有效容量
                effective_capacity = static_capacity / tau_factor

                # 计算当前车道的真实拥堵密度
                lane_density = n / effective_capacity
                lanes_density_sum += lane_density

            edge_density[id] = lanes_density_sum

        rest_road_density = [[edge_density[edge] for edge in road] for road in rest_road]

        # 计算路径的折扣密度和
        for i in range(len(rest_road_density)):
            gamma_sum = 0
            for index, value in enumerate(rest_road_density[i]):
                gamma_sum += value * pow(gamma, index)
            path_density.append(gamma_sum)

        in_lanes, out_lanes = self.get_allowed_movements(ts_id)
        current_lane = self.env.sumo_env.sumo.vehicle.getLaneID(veh_id)

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

    def get_all_vehicle_rouID_for_trip(self):
        ids = self.get_all_vehicle_id()
        curr_edge = {}
        if len(ids) > 0:
            for id in ids:
                curr_edge[id] = self.env.sumo_env.sumo.vehicle.getRoadID(id)

        return curr_edge

    def change_path(self, id, next_hops):
        self.env.sumo_env.sumo.vehicle.changeTarget(id, next_hops)

    def get_distance_to_intersection(self, veh_id):
        """
        获取车辆距离当前路段终点（路口/停止线）的剩余距离。

        :param veh_id: 车辆ID
        :return: 距离（米）。如果车辆不存在或在路口内部，返回 0.0 或 None。
        """
        try:
            # 1. 获取车辆当前所在的车道 ID
            lane_id = self.env.sumo_env.sumo.vehicle.getLaneID(veh_id)

            # 2. 如果 lane_id 为空，说明车辆可能还没进入路网
            if not lane_id:
                return None

            # 3. 如果是内部路段（以 ":" 开头），说明车辆已经在路口里面了
            # 此时距离路口（停止线）的距离在逻辑上是 0 或者负数
            if lane_id.startswith(":"):
                return 0.0

            # 4. 获取该车道的总长度
            lane_length = self.env.sumo_env.sumo.lane.getLength(lane_id)

            # 5. 获取车辆当前在车道上的位置 (从车道起点算起，到车头的位置)
            current_pos = self.env.sumo_env.sumo.vehicle.getLanePosition(veh_id)

            # 6. 计算剩余距离
            distance_to_end = lane_length - current_pos

            return distance_to_end

        except self.env.sumo_env.sumo.TraCIException:
            # 如果车辆ID错误或车辆刚刚离开仿真
            return None

    def judge_connection(self, edge_1, edge_2):

        lane_edge_1 = [lane for lane in self.env.sumo_env.sumo.lane.getIDList() if lane.startswith(edge_1)]
        for lane in lane_edge_1:
            links = self.env.sumo_env.sumo.lane.getLinks(lane)

            # links: [(via, toLane, direction, tls, state, tllink), ...]

            for link in links:
                to_lane = link[0]  # 下游 lane ID
                to_edge = self.env.sumo_env.sumo.lane.getEdgeID(to_lane)
                if to_edge == edge_2:
                    return True
        return False

    def get_all_obs(self):
        return self.env.sumo_env._compute_observations()

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

    def Onenter(self, v_id):
        lane_id = self.env.sumo_env.sumo.vehicle.getLaneID(v_id)
        if lane_id in self.enter_lane_list and len(self.env.sumo_env.sumo.vehicle.getRoute(v_id)) > 1:
            return True
        return False

    def _get_tls_by_lane(self, lane_id):
        for tls_id, lanes in self.enter_lane.items():
            if lane_id in lanes:
                return tls_id
        return None

    def _edge_free_flow_time(self, edge_id):
        if edge_id is None or str(edge_id).startswith(":"):
            return 0.0

        try:
            edge = self.topology.net.getEdge(edge_id)
            lanes = edge.getLanes()
            length = float(edge.getLength())
            speeds = [float(lane.getSpeed()) for lane in lanes if float(lane.getSpeed()) > 0.0]
            speed = max(speeds) if speeds else 11.11
            return length / max(speed, 1e-6)
        except Exception:
            pass

        try:
            lane_id = f"{edge_id}_0"
            length = float(self.env.sumo_env.sumo.lane.getLength(lane_id))
            speed = float(self.env.sumo_env.sumo.lane.getMaxSpeed(lane_id))
            return length / max(speed, 1e-6)
        except Exception:
            return 0.0

    def _route_free_flow_time(self, route):
        return float(sum(self._edge_free_flow_time(edge_id) for edge_id in route))

    def _get_vehicle_physical_edge(self, veh_id):
        try:
            if hasattr(self, "_get_current_physical_edge"):
                return self._get_current_physical_edge(veh_id)
            edge_id = self.env.sumo_env.sumo.vehicle.getRoadID(veh_id)
            if edge_id and not str(edge_id).startswith(":"):
                return edge_id
        except Exception:
            pass
        return None

    def _get_vehicle_free_flow_time(self, veh_id):
        try:
            route = self.env.sumo_env.sumo.vehicle.getRoute(veh_id)
            route_index = self.env.sumo_env.sumo.vehicle.getRouteIndex(veh_id)
        except Exception:
            return 0.0

        route_index = max(int(route_index), 0)
        remaining_route = route[route_index:] if route_index < len(route) else []
        completed_free_flow_time = self.veh_completed_free_flow_times.get(veh_id, 0.0)
        return completed_free_flow_time + self._route_free_flow_time(remaining_route)

    def _update_trip_delay_records(self):
        current_time = float(self.env.sumo_env.sumo.simulation.getTime())
        if self._last_trip_delay_update_time == current_time:
            return

        self._last_trip_delay_update_time = current_time
        self._pending_trip_delay_by_tls = {tls_id: 0.0 for tls_id in self.enter_lane}
        current_vehs = set(self.env.sumo_env.sumo.vehicle.getIDList())

        for veh_id in current_vehs:
            if veh_id not in self.veh_start_times:
                self.veh_start_times[veh_id] = current_time

            current_edge = self._get_vehicle_physical_edge(veh_id)
            if current_edge is not None:
                last_edge = self.veh_last_physical_edge.get(veh_id)
                if last_edge is None:
                    self.veh_last_physical_edge[veh_id] = current_edge
                elif current_edge != last_edge:
                    completed = self.veh_completed_free_flow_times.get(veh_id, 0.0)
                    self.veh_completed_free_flow_times[veh_id] = completed + self._edge_free_flow_time(last_edge)
                    self.veh_last_physical_edge[veh_id] = current_edge

            free_flow_time = self._get_vehicle_free_flow_time(veh_id)
            if free_flow_time > 0.0:
                self.veh_free_flow_times[veh_id] = free_flow_time

            if self.veh_origin_tls.get(veh_id) is None:
                try:
                    lane_id = self.env.sumo_env.sumo.vehicle.getLaneID(veh_id)
                    tls_id = self._get_tls_by_lane(lane_id)
                    if tls_id is not None:
                        self.veh_origin_tls[veh_id] = tls_id
                except Exception:
                    pass

        finished_vehs = [veh_id for veh_id in self.veh_start_times if veh_id not in current_vehs]
        for veh_id in finished_vehs:
            actual_travel_time = max(current_time - self.veh_start_times.get(veh_id, current_time), 0.0)
            free_flow_time = self.veh_free_flow_times.get(veh_id, 0.0)
            delay = max(float(actual_travel_time) - float(free_flow_time), 0.0)

            self.veh_travel_times.append(actual_travel_time)
            self.veh_delay_times.append(delay)

            tls_id = self.veh_origin_tls.get(veh_id)
            if tls_id in self._pending_trip_delay_by_tls:
                self._pending_trip_delay_by_tls[tls_id] += delay
                self.veh_delay_sums_by_tls[tls_id] += delay
                self.veh_delay_counts_by_tls[tls_id] += 1
            elif len(self._pending_trip_delay_by_tls) > 0:
                share = delay / len(self._pending_trip_delay_by_tls)
                for key in self._pending_trip_delay_by_tls:
                    self._pending_trip_delay_by_tls[key] += share

            self.veh_start_times.pop(veh_id, None)
            self.veh_free_flow_times.pop(veh_id, None)
            self.veh_completed_free_flow_times.pop(veh_id, None)
            self.veh_last_physical_edge.pop(veh_id, None)
            self.veh_origin_tls.pop(veh_id, None)

    def get_delay(self):
        self._update_trip_delay_records()
        return self._pending_trip_delay_by_tls.copy()

    def set_traffic_accident(self, vehicle_id, edge_id, pos, duration):
        id_list = self.get_all_vehicle_id()

        if vehicle_id not in id_list and self.flag_accident == 1:
            self.flag_accident = 0
        if vehicle_id in id_list and self.flag_accident == 0:
            self.env.sumo_env.sumo.vehicle.setStop(vehID=vehicle_id, edgeID=edge_id, pos=pos, duration=duration)
            self.flag_accident = 1

    def get_vehicle_list(self):
        return self.env.sumo_env.sumo.vehicle.getIDList()

    def get_vehicle_speed(self):
        vehicle_ids = self.env.sumo_env.sumo.vehicle.getIDList()
        speed_dict = {}
        for vid in vehicle_ids:
            speed = self.env.sumo_env.sumo.vehicle.getSpeed(vid)
            speed_dict[vid] = speed
        return speed_dict

    def update_travel_times(self):
        self._update_trip_delay_records()

    def get_mean_travel_time(self):
        """回合结束时调用，获取所有完成行程车辆的平均旅行时间"""
        if len(self.veh_travel_times) == 0:
            return 0.0
        return float(np.mean(self.veh_travel_times))

    def get_mean_delay(self):
        self._update_trip_delay_records()
        if len(self.veh_delay_times) == 0:
            return 0.0
        return float(np.mean(self.veh_delay_times))

    def get_total_delay(self):
        self._update_trip_delay_records()
        if len(self.veh_delay_times) == 0:
            return 0.0
        return float(np.sum(self.veh_delay_times))

    def get_completed_vehicle_count(self):
        self._update_trip_delay_records()
        return len(self.veh_travel_times)
