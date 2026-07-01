
import xml.etree.ElementTree as ET

import sumolib
import networkx as nx
import traci


class RoadNetTopology:
    def __init__(self, net_file_path):
        """
        初始化：读取 SUMO 路网文件，并构建纯内存的有向图。
        图中的节点是物理 Edge，不包含 ':' 开头的内部连接段。
        """
        print(f"正在构建路网拓扑图: {net_file_path} ...")
        self.net = sumolib.net.readNet(net_file_path)
        self.G = nx.DiGraph()
        self._build_graph()
        print(f"拓扑图构建完成！包含 {self.G.number_of_nodes()} 个路段(Edges) 和 {self.G.number_of_edges()} 条连接。")

    def _build_graph(self):
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue

            self.G.add_node(edge_id)
            for to_edge in edge.getOutgoing():
                to_edge_id = to_edge.getID()
                if not to_edge_id.startswith(":"):
                    self.G.add_edge(edge_id, to_edge_id)

    def _build_subgraph_without_forbidden(self, forbidden_edges):
        if not forbidden_edges:
            return self.G
        keep_nodes = [n for n in self.G.nodes if n not in forbidden_edges]
        return self.G.subgraph(keep_nodes).copy()

    def get_distance_to_dest(self, edge, destination_edge):
        """获取从edge到destination_edge的最短路径长度（跳数）"""
        if edge == destination_edge:
            return 0
        if edge not in self.G or destination_edge not in self.G:
            return float('inf')
        try:
            return nx.shortest_path_length(self.G, edge, destination_edge)
        except nx.NetworkXNoPath:
            return float('inf')

    def get_valid_next_hops_fast(self, current_edge, destination_edge, forbidden_edges=None):
        """
        返回合法下一跳列表。
        过滤原则：
        1. 必须和 current_edge 拓扑相连；
        2. 必须仍然可达 destination_edge；
        3. 允许最多比当前最短路多 1 跳，给 RL 留探索空间（而非严格递减）；
        4. 可选地排除 recent forbidden edges，避免车辆在网内来回绕圈。
        """
        if current_edge == destination_edge:
            return []

        if current_edge not in self.G or destination_edge not in self.G:
            return []

        forbidden_edges = set(forbidden_edges or [])
        forbidden_edges.discard(current_edge)
        forbidden_edges.discard(destination_edge)

        try:
            current_dist = nx.shortest_path_length(self.G, current_edge, destination_edge)
        except nx.NetworkXNoPath:
            return []

        candidates = []
        for next_edge in sorted(self.G.successors(current_edge)):
            if next_edge in forbidden_edges and next_edge != destination_edge:
                continue
            try:
                next_dist = nx.shortest_path_length(self.G, next_edge, destination_edge)
            except nx.NetworkXNoPath:
                continue

            # ===== 修复: 允许绕路1跳, 给RL探索空间 =====
            # 严格 <= 会导致很多路口只有1个选择, RL没有决策空间
            if next_dist <= current_dist + 1:
                candidates.append((next_dist, next_edge))

        candidates.sort(key=lambda x: (x[0], x[1]))
        return [edge for _, edge in candidates]

    def build_route_with_next_hop(self, current_edge, chosen_next_edge, destination_edge, forbidden_edges=None):
        """
        先固定 chosen_next_edge，再从 chosen_next_edge 到真实终点重算一条无环尾路径。
        若去除 forbidden edges 后无路可走，则回退到原图最短路。
        """
        if current_edge not in self.G or chosen_next_edge not in self.G or destination_edge not in self.G:
            return None

        forbidden_edges = set(forbidden_edges or [])
        forbidden_edges.discard(current_edge)
        forbidden_edges.discard(chosen_next_edge)
        forbidden_edges.discard(destination_edge)

        graph = self._build_subgraph_without_forbidden(forbidden_edges)
        try:
            tail = nx.shortest_path(graph, chosen_next_edge, destination_edge)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                tail = nx.shortest_path(self.G, chosen_next_edge, destination_edge)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        if not tail:
            return None
        return [current_edge] + tail



def get_all_vehicle_route_lengths(xml_filepath):

    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # 1. 提取预定义的 route
    predefined_routes = {}
    for route in root.findall('route'):
        route_id = route.get('id')
        edges_str = route.get('edges', '')
        if route_id and edges_str:
            predefined_routes[route_id] = len(edges_str.split())

    vehicle_lengths = {}

    # 2. 遍历所有 vehicle 标签，获取每辆车的 edge 数量
    for vehicle in root.findall('vehicle'):
        veh_id = vehicle.get('id')
        edges_count = 0

        # 格式 A: 嵌套的 <route>
        child_route = vehicle.find('route')
        if child_route is not None:
            edges_str = child_route.get('edges', '')
            edges_count = len(edges_str.split())

        # 格式 B: 引用的 route 属性
        elif vehicle.get('route'):
            route_id = vehicle.get('route')
            edges_count = predefined_routes.get(route_id, 0)

        vehicle_lengths[veh_id] = edges_count

    return vehicle_lengths




def find_longest_route(xml_filepath):

    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # 1. 提取预定义的 route (如果文件中有 <route id="..." edges="..."/>)
    predefined_routes = {}
    for route in root.findall('route'):
        route_id = route.get('id')
        edges_str = route.get('edges', '')
        if route_id and edges_str:
            predefined_routes[route_id] = edges_str.split()

    max_edges_count = 0
    longest_veh_id = None
    longest_edges_list = []

    # 2. 遍历所有 vehicle 标签
    for vehicle in root.findall('vehicle'):
        veh_id = vehicle.get('id')
        edges_list = []

        # 格式 A: <vehicle> 内部嵌套了 <route edges="..."/>
        child_route = vehicle.find('route')
        if child_route is not None:
            edges_str = child_route.get('edges', '')
            edges_list = edges_str.split()

        # 格式 B: <vehicle route="route_id" .../> 引用了外部预定义的 route
        elif vehicle.get('route'):
            route_id = vehicle.get('route')
            edges_list = predefined_routes.get(route_id, [])

        # 3. 对比并记录最大值
        current_edges_count = len(edges_list)
        if current_edges_count > max_edges_count:
            max_edges_count = current_edges_count
            longest_veh_id = veh_id
            longest_edges_list = edges_list

    return longest_veh_id, max_edges_count, longest_edges_list




def get_agent_node_id(network, agent_names):
    nodes = network.getNodes()
    nodes_dict = {node.getID(): node for node in nodes}
    agent_nodes = {agent_name: nodes_dict[agent_name] for agent_name in agent_names}
    return list(agent_nodes.values())



def get_junction_distance(network, from_to):
    edges = network.getEdges()
    agent_edge = {}

    for edge in edges:
        edge_id = edge.getFromNode().getID() + edge.getToNode().getID()
        if edge_id in from_to:
            agent_edge[edge_id] = edge.getLength()
    return agent_edge



def get_neighbor_edge_index(network, agent_names):
    nodes = get_agent_node_id(network, agent_names)
    agent_names_number = []
    for i in range(len(agent_names)):
        agent_names_number.append(i)

    start = []
    end = []
    from_to = []
    start_name = []
    end_name = []
    for node in nodes:
        node_id = node.getID()
        incoming_edges = node.getIncoming()
        for edge in incoming_edges:
            connected_node = edge.getFromNode()
            if connected_node.getID() in agent_names:
                start.append(agent_names_number[agent_names.index(connected_node.getID())])
                end.append(agent_names_number[agent_names.index(node_id)])
                from_to.append(connected_node.getID() + node_id)
                start_name.append(connected_node.getID())
                end_name.append(node_id)

    edge_index = [start, end]
    edge_index_name = [start_name, end_name]
    return edge_index, from_to, edge_index_name



def get_influence_infos(env, from_to, agent_names, weight_length, edge_index_name):
    phase_id = env.get_phase()
    lanes_density, out_lanes_density = env.get_vehicle_number()
    influence_infos = {}
    for i in range(len(from_to)):
        f = edge_index_name[0][i]
        to = edge_index_name[1][i]
        f_id = agent_names.index(f)
        to_id = agent_names.index(to)
        influence_infos[from_to[i]] = [f_id, to_id, weight_length[from_to[i]]] + phase_id[to] + lanes_density[f]
    return influence_infos



def change_edge_index(influence_coeff, edge_index, from_to):
    from_to_ = from_to.copy()
    for ft in from_to:

        ft_index = from_to_.index(ft)
        if influence_coeff[ft] < 6:
            edge_index[0].pop(ft_index)
            edge_index[1].pop(ft_index)
            from_to_.pop(ft_index)
    return edge_index



def get_new_edge_infos(env, from_to, agent_names, weight_length, agents, edge_index, edge_index_name):
    influence_infos = get_influence_infos(env, from_to, agent_names, weight_length, edge_index_name)
    influence_coeff = {ft: agents.choose_action(influence_infos[ft]) for ft in from_to}
    new_edge_index = change_edge_index(influence_coeff, edge_index, from_to)
    return influence_infos, influence_coeff, new_edge_index



def get_new_edge_infos_(env, from_to, agent_names, weight_length, agents, edge_index, edge_index_name):
    influence_infos = get_influence_infos(env, from_to, agent_names, weight_length, edge_index_name)
    influence_coeff = {ft: agents[ft].choose_action(influence_infos[ft]) for ft in from_to}
    new_edge_index = change_edge_index(influence_coeff, edge_index, from_to)
    return influence_infos, influence_coeff, new_edge_index



def get_graph(net):
    edge_dict = {}
    for edge in net.getEdges():
        edge_id = edge.getID()
        outgoing = [e.getID() for e in edge.getOutgoing()]
        edge_dict[edge_id] = outgoing
    return edge_dict


# BFS查找所有路径（简单例子）
def bfs_all_paths(net, start, goal, max_depth=10):
    from collections import deque
    graph = get_graph(net)
    queue = deque([[start]])
    paths = []
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == goal:
            paths.append(path)
        elif len(path) < max_depth:
            for neighbor in graph.get(current, []):
                if neighbor not in path:
                    queue.append(path + [neighbor])
    return paths



def find_next(current_edge, all_path):
    paths = all_path[:20]
    next_hops = set()
    rest_road = []
    for path in paths:
        if current_edge in path:
            current_index = path.index(current_edge)
            if current_index + 1 < len(path):
                next_hops.add(path[current_index + 1])
                rest_road.append(path[current_index + 1:])

    return next_hops, rest_road


def extract_vehicle_routes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    vehicle_routes = {}
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        route = vehicle.find('route')
        if route is not None:
            edges = route.get('edges').split()
            start_edge = edges[0]
            end_edge = edges[-1]
            vehicle_routes[vid] = [start_edge, end_edge]

    return vehicle_routes
