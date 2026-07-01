from env.wrapper_env import wrapper_env_sumo
import numpy as np
from until import RoadNetTopology, extract_vehicle_routes


class Vehicle_single_env(wrapper_env_sumo):

    def __init__(self, **kwargs):
        super(Vehicle_single_env, self).__init__(**kwargs)
        self.topology = RoadNetTopology(kwargs['net_file'])

        self.all_vehicle_start_end = extract_vehicle_routes(kwargs['route_file'])
        self.all_vehicle_target_road = {k: v[1] for k, v in self.all_vehicle_start_end.items()}

        self.all_vehicle_target_pos = {}
        for vid, target_edge_id in self.all_vehicle_target_road.items():
            try:
                edge = self.topology.net.getEdge(target_edge_id)
                self.all_vehicle_target_pos[vid] = edge.getShape()[0]
            except KeyError:
                self.all_vehicle_target_pos[vid] = (0.0, 0.0)

        self.prev_driving_distance = {}
        self.reward_prev_edge = {}
        self.prev_physical_edge = {}
        self.edge_density_ema = {}
        self.vehicle_edge_history = {}
        self.vehicle_edge_visit_count = {}
        self.last_decision_edge = {}
        self.current_veh_states = {}
        self.controlled_vehicle_ids = None

        self.decision_trigger_distance = 100.0
        self.segment_reward_bonus = 5.0
        self.max_action_dim = 3
        self.loop_history_window = 6

    def init_infos(self):
        super().init_infos()

    def set_controlled_vehicle_ids(self, vehicle_ids):
        self.controlled_vehicle_ids = None if vehicle_ids is None else set(vehicle_ids)

    def _is_controlled_vehicle(self, veh_id):
        return self.controlled_vehicle_ids is None or veh_id in self.controlled_vehicle_ids

    def _iter_controlled_vehicle_ids(self, vehicle_ids):
        if self.controlled_vehicle_ids is None:
            return list(vehicle_ids)
        return [veh_id for veh_id in vehicle_ids if veh_id in self.controlled_vehicle_ids]

    def _clear_vehicle_trackers(self, active_vehicle_ids):
        active_vehicle_ids = set(active_vehicle_ids)
        for store in [
            self.prev_driving_distance,
            self.reward_prev_edge,
            self.prev_physical_edge,
            self.vehicle_edge_history,
            self.vehicle_edge_visit_count,
            self.last_decision_edge,
        ]:
            stale_ids = [veh_id for veh_id in store.keys() if veh_id not in active_vehicle_ids]
            for veh_id in stale_ids:
                store.pop(veh_id, None)

    def _get_current_physical_edge(self, veh_id, raw_road_id=None):
        if raw_road_id is None:
            raw_road_id = self.env.sumo_env.sumo.vehicle.getRoadID(veh_id)

        if raw_road_id and not str(raw_road_id).startswith(":"):
            return raw_road_id

        prev_edge = self.prev_physical_edge.get(veh_id)
        if prev_edge and not str(prev_edge).startswith(":"):
            return prev_edge

        try:
            route = self.env.sumo_env.sumo.vehicle.getRoute(veh_id)
            route_idx = self.env.sumo_env.sumo.vehicle.getRouteIndex(veh_id)
            candidate_indices = [route_idx, route_idx - 1, route_idx + 1]
            for idx in candidate_indices:
                if 0 <= idx < len(route) and not route[idx].startswith(":"):
                    return route[idx]
        except Exception:
            pass

        return raw_road_id

    def _get_recent_forbidden_edges(self, veh_id, current_edge=None, chosen_next_edge=None,
                                    destination_edge=None, relax=False):
        """
        获取禁止边集合。
        relax=True 时放宽约束（仅禁止访问3次以上的边），用于候选不足时的回退。
        """
        history = self.vehicle_edge_history.get(veh_id, [])
        visit_counter = self.vehicle_edge_visit_count.get(veh_id, {})

        if relax:
            # 放宽模式：只禁止重度循环的边（>=3次）
            forbidden_edges = {
                edge_id for edge_id, cnt in visit_counter.items() if cnt >= 3
            }
        else:
            # 正常模式
            recent_edges = set(history[-self.loop_history_window:])
            heavy_revisit_edges = {
                edge_id for edge_id, cnt in visit_counter.items() if cnt >= 2
            }
            forbidden_edges = recent_edges | heavy_revisit_edges

        # 永远不禁止当前边、下一跳、终点
        for edge_id in [current_edge, chosen_next_edge, destination_edge]:
            if edge_id is not None:
                forbidden_edges.discard(edge_id)
        return forbidden_edges

    def _update_vehicle_edge_history(self, veh_states):
        active_vehicle_ids = set(veh_states.keys())
        for veh_id, state in veh_states.items():
            edge_id = state.get('physical_edge')
            if not edge_id or str(edge_id).startswith(":"):
                continue

            prev_edge = self.prev_physical_edge.get(veh_id)
            if prev_edge != edge_id:
                self.vehicle_edge_history.setdefault(veh_id, []).append(edge_id)
                self.vehicle_edge_history[veh_id] = self.vehicle_edge_history[veh_id][-20:]
                visit_counter = self.vehicle_edge_visit_count.setdefault(veh_id, {})
                visit_counter[edge_id] = visit_counter.get(edge_id, 0) + 1
                self.last_decision_edge.pop(veh_id, None)

            self.prev_physical_edge[veh_id] = edge_id

        self._clear_vehicle_trackers(active_vehicle_ids)

    def _get_all_vehicles_state(self):
        veh_states = {}
        vehicle_ids = self._iter_controlled_vehicle_ids(self.env.sumo_env.sumo.vehicle.getIDList())

        for veh_id in vehicle_ids:
            road_id = self.env.sumo_env.sumo.vehicle.getRoadID(veh_id)
            physical_edge = self._get_current_physical_edge(veh_id, road_id)
            route = self.env.sumo_env.sumo.vehicle.getRoute(veh_id)
            final_route_edge = route[-1] if len(route) > 0 else physical_edge
            driving_distance = self.env.sumo_env.sumo.vehicle.getDrivingDistance(veh_id, final_route_edge, 0.0)

            veh_states[veh_id] = {
                'speed': self.env.sumo_env.sumo.vehicle.getSpeed(veh_id),
                'acceleration': self.env.sumo_env.sumo.vehicle.getAcceleration(veh_id),
                'position': self.env.sumo_env.sumo.vehicle.getPosition(veh_id),
                'road_id': road_id,
                'physical_edge': physical_edge,
                'lane_id': self.env.sumo_env.sumo.vehicle.getLaneID(veh_id),
                'lane_pos': self.env.sumo_env.sumo.vehicle.getLanePosition(veh_id),
                'route_id': self.env.sumo_env.sumo.vehicle.getRouteID(veh_id),
                'waiting_time': self.env.sumo_env.sumo.vehicle.getWaitingTime(veh_id),
                'leader_dist': self.env.sumo_env.sumo.vehicle.getLeader(veh_id),
                'next_TSC': self.env.sumo_env.sumo.vehicle.getNextTLS(veh_id),
                'max_speed': self.env.sumo_env.sumo.vehicle.getAllowedSpeed(veh_id),
                'best_lanes': self.env.sumo_env.sumo.vehicle.getBestLanes(veh_id),
                'getRoute': route,
                'getLeftLeaders': self.env.sumo_env.sumo.vehicle.getLeftLeaders(veh_id),
                'getRightLeaders': self.env.sumo_env.sumo.vehicle.getRightLeaders(veh_id),
                'getDrivingDistance': driving_distance,
                'AccumulatedWaitingTime': self.env.sumo_env.sumo.vehicle.getAccumulatedWaitingTime(veh_id),
            }

        return veh_states

    def reset(self, seed=None, options=None):
        obs = super().reset()

        self.prev_driving_distance = {}
        self.reward_prev_edge = {}
        self.prev_physical_edge = {}
        self.edge_density_ema.clear()
        self.vehicle_edge_history = {}
        self.vehicle_edge_visit_count = {}
        self.last_decision_edge = {}

        for tls_id in obs.keys():
            extra_obs = self.get_50m_direction_counts_normalized(tls_id)
            obs[tls_id] = np.concatenate([obs[tls_id], extra_obs], axis=0).astype(np.float32)

        self.current_veh_states = self._get_all_vehicles_state()
        self._update_vehicle_edge_history(self.current_veh_states)

        veh_obs, action_masks, next_hops = self.process_vehicle_states(self.current_veh_states)
        decision_vehicle_ids = self.get_decision_vehicle_ids(self.current_veh_states, action_masks, next_hops)
        info = {
            'veh_obs': veh_obs,
            'action_masks': action_masks,
            'next_hops': next_hops,
            'decision_vehicle_ids': decision_vehicle_ids,
        }
        return obs, info

    def get_decision_vehicle_ids(self, raw_veh_dict, action_masks=None, next_hops=None):
        decision_vehicle_ids = []
        for veh_id, raw_info in raw_veh_dict.items():
            if not self._is_controlled_vehicle(veh_id):
                continue

            current_edge = raw_info.get('physical_edge')
            if (not current_edge) or str(current_edge).startswith(":"):
                continue

            if current_edge == self.all_vehicle_target_road.get(veh_id):
                continue

            next_tls = raw_info.get('next_TSC', [])
            if len(next_tls) == 0:
                continue

            dist_to_stop_line = next_tls[0][2]
            if dist_to_stop_line is None or dist_to_stop_line <= 0.0 or dist_to_stop_line > self.decision_trigger_distance:
                continue

            if self.last_decision_edge.get(veh_id) == current_edge:
                continue

            if action_masks is not None and np.sum(action_masks.get(veh_id, np.zeros(self.max_action_dim))) <= 0:
                continue

            if next_hops is not None and len(next_hops.get(veh_id, [])) == 0:
                continue

            decision_vehicle_ids.append(veh_id)

        return decision_vehicle_ids

    def step(self, actions):
        single_actions, route_actions = actions
        alive_vehicles = set(self.env.sumo_env.sumo.vehicle.getIDList())

        for veh_id, chosen_next_edge in route_actions.items():
            if veh_id not in alive_vehicles or chosen_next_edge is None:
                continue

            try:
                next_tls = self.env.sumo_env.sumo.vehicle.getNextTLS(veh_id)
                if len(next_tls) == 0:
                    continue
                dist_to_stop_line = next_tls[0][2]
                if dist_to_stop_line is None or dist_to_stop_line <= 0.0 or dist_to_stop_line > self.decision_trigger_distance:
                    continue

                current_edge = self._get_current_physical_edge(veh_id)
                # ===== 修复: 不再重新计算mask验证, 直接信任决策时刻的选择 =====
                changed = self._apply_route_change(veh_id, chosen_next_edge)
                if changed:
                    self.last_decision_edge[veh_id] = current_edge
            except Exception:
                continue

        next_obs, reward, done, info = super(Vehicle_single_env, self).step(single_actions)

        for tls_id in next_obs.keys():
            extra_obs = self.get_50m_direction_counts_normalized(tls_id)
            next_obs[tls_id] = np.concatenate([next_obs[tls_id], extra_obs], axis=0).astype(np.float32)

        arrived_vehicles = self.env.sumo_env.sumo.simulation.getArrivedIDList()

        self.current_veh_states = self._get_all_vehicles_state()
        self._update_vehicle_edge_history(self.current_veh_states)
        veh_reward = self.get_vehicle_rewards(arrived_vehicles)

        next_veh_obs, next_action_masks, next_next_hops = self.process_vehicle_states(self.current_veh_states)
        decision_vehicle_ids = self.get_decision_vehicle_ids(self.current_veh_states, next_action_masks, next_next_hops)

        info['veh_obs'] = next_veh_obs
        info['action_masks'] = next_action_masks
        info['next_hops'] = next_next_hops
        info['veh_reward'] = veh_reward
        info['decision_vehicle_ids'] = decision_vehicle_ids
        return next_obs, reward, done, info

    def _apply_route_change(self, veh_id, chosen_next_edge):
        """直接应用路径变更，不再二次验证mask（信任决策时刻的结果）"""
        final_edge = self.all_vehicle_target_road.get(veh_id)
        current_edge = self._get_current_physical_edge(veh_id)
        if current_edge is None or final_edge is None:
            return False

        forbidden_edges = self._get_recent_forbidden_edges(
            veh_id,
            current_edge=current_edge,
            chosen_next_edge=chosen_next_edge,
            destination_edge=final_edge,
        )
        new_route = self.topology.build_route_with_next_hop(
            current_edge=current_edge,
            chosen_next_edge=chosen_next_edge,
            destination_edge=final_edge,
            forbidden_edges=forbidden_edges,
        )
        if not new_route or len(new_route) < 2:
            return False

        try:
            self.env.sumo_env.sumo.vehicle.setRoute(veh_id, new_route)
            return True
        except Exception:
            return False

    def _get_edge_reachable_edges(self, veh_id):
        reachable_edges = set()
        try:
            edge_id = self._get_current_physical_edge(veh_id)
            if (not edge_id) or str(edge_id).startswith(":"):
                return reachable_edges

            lane_count = self.env.sumo_env.sumo.edge.getLaneNumber(edge_id)
            for i in range(lane_count):
                lane_id = f"{edge_id}_{i}"
                links = self.env.sumo_env.sumo.lane.getLinks(lane_id)
                for link in links:
                    to_lane = link[0]
                    if not to_lane:
                        continue
                    to_edge = self.env.sumo_env.sumo.lane.getEdgeID(to_lane)
                    if not to_edge.startswith(":"):
                        reachable_edges.add(to_edge)
        except Exception:
            pass

        return reachable_edges

    def get_action_mask_for_vehicle(self, veh_id, destination_edge_id):
        """
        生成action mask。
        关键设计原则:
        1. 按拓扑距离排序 (不使用visit_count), 保证动作语义稳定
        2. 候选不足时自动放宽forbidden约束
        3. 允许轻微绕路 (distance + 1), 给RL探索空间
        """
        current_edge = self._get_current_physical_edge(veh_id)

        if current_edge is None or str(current_edge).startswith(":"):
            return current_edge, [], np.zeros(self.max_action_dim, dtype=np.int8)

        forbidden_edges = self._get_recent_forbidden_edges(
            veh_id,
            current_edge=current_edge,
            destination_edge=destination_edge_id,
        )

        topo_next_edges = self.topology.get_valid_next_hops_fast(
            current_edge,
            destination_edge_id,
            forbidden_edges=forbidden_edges,
        )

        edge_reachable_edges = self._get_edge_reachable_edges(veh_id)

        valid_next_edges = [e for e in topo_next_edges if e in edge_reachable_edges and e != current_edge]

        # ===== 修复: 候选不足时, 放宽forbidden约束再试一次 =====
        if len(valid_next_edges) < 2:
            relaxed_forbidden = self._get_recent_forbidden_edges(
                veh_id,
                current_edge=current_edge,
                destination_edge=destination_edge_id,
                relax=True,
            )
            relaxed_topo = self.topology.get_valid_next_hops_fast(
                current_edge,
                destination_edge_id,
                forbidden_edges=relaxed_forbidden,
            )
            relaxed_valid = [e for e in relaxed_topo if e in edge_reachable_edges and e != current_edge]
            # 合并: 原来的优先, 补充新的
            existing = set(valid_next_edges)
            for e in relaxed_valid:
                if e not in existing:
                    valid_next_edges.append(e)

        if destination_edge_id in valid_next_edges:
            valid_next_edges = [destination_edge_id]

        # ===== 修复: 只按拓扑距离排序, 不用visit_count =====
        # 动作语义固定: action 0 = 最短路方向, action 1 = 次短, action 2 = 绕路方向
        # 这样Q网络可以学到稳定的"action 0通常最优, 但拥堵时选action 1/2"
        def _sort_key(e):
            try:
                dist = self.topology.get_distance_to_dest(e, destination_edge_id)
            except Exception:
                dist = float('inf')
            return (dist, e)  # 距离 + edge_id (确定性排序)

        valid_next_edges.sort(key=_sort_key)
        valid_next_edges = valid_next_edges[:self.max_action_dim]

        action_mask = np.zeros(self.max_action_dim, dtype=np.int8)
        if len(valid_next_edges) > 0:
            action_mask[:len(valid_next_edges)] = 1
        # print(action_mask,'action_mask')
        return current_edge, valid_next_edges, action_mask

    def process_vehicle_states(self, raw_veh_dict):
        rl_states_dict = {}
        action_masks = {}
        next_hops = {}

        raw_veh_dict = {
            veh_id: raw_info
            for veh_id, raw_info in raw_veh_dict.items()
            if self._is_controlled_vehicle(veh_id)
        }
        if not raw_veh_dict:
            return rl_states_dict, action_masks, next_hops

        raw_out_lanes_density = self.get_out_lanes_for_dict()
        current_edge_densities = {}
        for _, lane_dict in raw_out_lanes_density.items():
            for lane_id, density in lane_dict.items():
                edge_id = lane_id.rsplit('_', 1)[0]
                current_edge_densities.setdefault(edge_id, []).append(density)

        for edge_id in current_edge_densities:
            current_edge_densities[edge_id] = np.mean(current_edge_densities[edge_id])

        NORM_CONFIG = {
            'speed': (0.0, 11.111),
            'AccumulatedWaitingTime': (0.0, 1000.0),
            'leader_dist': (0.0, 100.0),
            'next_TSC_distance': (0.0, 800.0),
            'getDrivingDistance': (0.0, 5000.0),
        }

        def _normalize(val, min_val, max_val):
            normalized = (val - min_val) / (max_val - min_val)
            return np.clip(normalized, 0.0, 1.0)

        for veh_id, raw_info in raw_veh_dict.items():
            speed = raw_info['speed']
            pos_x, pos_y = raw_info['position']
            driving_distance = raw_info['getDrivingDistance']
            accumulated_waiting_time = raw_info['AccumulatedWaitingTime']

            leader_dist = raw_info['leader_dist']
            leader_dist = 100.0 if leader_dist is None else leader_dist[-1]

            if len(raw_info['next_TSC']) == 0:
                next_tsc_distance = 800.0
                next_tsc_state = [1.0]
            else:
                next_tsc = raw_info['next_TSC'][0]
                next_tsc_distance = next_tsc[2]
                phase_char = next_tsc[3].lower()
                if phase_char == 'g':
                    next_tsc_state = [1.0]
                elif phase_char == 'y':
                    next_tsc_state = [0.5]
                else:
                    next_tsc_state = [0.0]

            target_x, target_y = self.all_vehicle_target_pos.get(veh_id, (pos_x, pos_y))
            dx = np.clip((target_x - pos_x) / 5600.0, -1.0, 1.0)
            dy = np.clip((target_y - pos_y) / 5600.0, -1.0, 1.0)

            _, next_hop, action_mask = self.get_action_mask_for_vehicle(
                veh_id,
                self.all_vehicle_target_road.get(veh_id),
            )
            action_masks[veh_id] = action_mask
            next_hops[veh_id] = next_hop

            downstream_densities = np.zeros(self.max_action_dim, dtype=np.float32)
            alpha = 0.3
            for i, edge in enumerate(next_hop[:self.max_action_dim]):
                instant_density = current_edge_densities.get(edge, 0.0)
                if edge not in self.edge_density_ema:
                    self.edge_density_ema[edge] = instant_density
                else:
                    self.edge_density_ema[edge] = alpha * instant_density + (1 - alpha) * self.edge_density_ema[edge]
                downstream_densities[i] = np.clip(self.edge_density_ema[edge], 0.0, 1.0)

            # ===== 修复: 增加有效动作数量到state中 =====
            # 让Q网络知道当前有几个合法选项
            n_valid_actions = float(np.sum(action_mask)) / self.max_action_dim  # 归一化到[0,1]

            state_vector = np.array([
                _normalize(speed, *NORM_CONFIG['speed']),
                _normalize(driving_distance, *NORM_CONFIG['getDrivingDistance']),
                _normalize(accumulated_waiting_time, *NORM_CONFIG['AccumulatedWaitingTime']),
                _normalize(next_tsc_distance, *NORM_CONFIG['next_TSC_distance']),
                _normalize(leader_dist, *NORM_CONFIG['leader_dist']),
                dx,
                dy,
            ], dtype=np.float32)

            state_vector = np.concatenate([
                state_vector,
                np.array(next_tsc_state, dtype=np.float32),
                downstream_densities,
                np.array([n_valid_actions], dtype=np.float32),
            ])
            rl_states_dict[veh_id] = state_vector

        return rl_states_dict, action_masks, next_hops

    def get_vehicle_rewards(self, arrived_vehicles):
        veh_rewards = {}
        ARRIVAL_BONUS = 20.0
        STEP_PENALTY = -0.1
        WEIGHT_WAITING = -0.05
        WEIGHT_DISTANCE = 0.1
        SEGMENT_BONUS = 1.0

        for veh_id, state in self.current_veh_states.items():
            if not self._is_controlled_vehicle(veh_id):
                continue

            reward = STEP_PENALTY

            if state['speed'] < 0.1:
                reward += WEIGHT_WAITING

            current_dist = state['getDrivingDistance']
            prev_dist = self.prev_driving_distance.get(veh_id, current_dist)
            if current_dist > 0 and prev_dist > 0:
                dist_reduction = np.clip(prev_dist - current_dist, -20.0, 20.0)
                reward += WEIGHT_DISTANCE * dist_reduction
            self.prev_driving_distance[veh_id] = current_dist

            current_edge = state.get('physical_edge')
            prev_edge = self.reward_prev_edge.get(veh_id)
            if prev_edge is not None and current_edge is not None and current_edge != prev_edge:
                reward += SEGMENT_BONUS
            if current_edge is not None:
                self.reward_prev_edge[veh_id] = current_edge

            veh_rewards[veh_id] = reward

        for veh_id in arrived_vehicles:
            if not self._is_controlled_vehicle(veh_id):
                continue

            veh_rewards[veh_id] = veh_rewards.get(veh_id, 0.0) + ARRIVAL_BONUS
            self.prev_driving_distance.pop(veh_id, None)
            self.reward_prev_edge.pop(veh_id, None)
            self.prev_physical_edge.pop(veh_id, None)
            self.vehicle_edge_history.pop(veh_id, None)
            self.vehicle_edge_visit_count.pop(veh_id, None)
            self.last_decision_edge.pop(veh_id, None)

        return veh_rewards

    def get_50m_direction_counts_normalized(self, tls_id):
        controlled_lanes = list(set(self.env.sumo_env.sumo.trafficlight.getControlledLanes(tls_id)))

        straight_count, left_count, right_count = 0, 0, 0
        straight_lanes, left_lanes, right_lanes = 0, 0, 0

        for lane in controlled_lanes:
            lane_length = self.env.sumo_env.sumo.lane.getLength(lane)
            links = self.env.sumo_env.sumo.lane.getLinks(lane)
            allowed_dirs = [link[6].lower() for link in links if len(link) > 6]

            if 's' in allowed_dirs:
                straight_lanes += 1
            if 'l' in allowed_dirs or 'L' in allowed_dirs or 't' in allowed_dirs:
                left_lanes += 1
            if 'r' in allowed_dirs or 'R' in allowed_dirs:
                right_lanes += 1

            veh_ids = self.env.sumo_env.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in veh_ids:
                pos = self.env.sumo_env.sumo.vehicle.getLanePosition(vid)
                if (lane_length - pos) <= 50.0:
                    if 's' in allowed_dirs:
                        straight_count += 1
                    if 'l' in allowed_dirs or 'L' in allowed_dirs or 't' in allowed_dirs:
                        left_count += 1
                    if 'r' in allowed_dirs or 'R' in allowed_dirs:
                        right_count += 1

        l_veh = 5.0
        g_min = 2.5
        max_per_lane = 50.0 / (l_veh + g_min)

        s_norm = straight_count / (straight_lanes * max_per_lane) if straight_lanes > 0 else 0.0
        l_norm = left_count / (left_lanes * max_per_lane) if left_lanes > 0 else 0.0
        r_norm = right_count / (right_lanes * max_per_lane) if right_lanes > 0 else 0.0

        return [np.clip(s_norm, 0.0, 1.0), np.clip(l_norm, 0.0, 1.0), np.clip(r_norm, 0.0, 1.0)]
