import xml.etree.ElementTree as ET


def get_agent_node_id(network, agent_names):
    nodes = network.getNodes()
    # print(nodes)
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
        incoming_edges = node.getIncoming()  # 使用 getIncoming() 方法
        for edge in incoming_edges:
            connected_node = edge.getFromNode()  # 获取连接的源路口
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
    # 获取每个路口相邻路口的会对本路口造成影响的信息
    influence_infos = get_influence_infos(env, from_to, agent_names, weight_length, edge_index_name)
    influence_coeff = {ft: agents.choose_action(influence_infos[ft]) for ft in from_to}
    new_edge_index = change_edge_index(influence_coeff, edge_index, from_to)
    return influence_infos, influence_coeff, new_edge_index


def get_new_edge_infos_(env, from_to, agent_names, weight_length, agents, edge_index, edge_index_name):
    # 获取每个路口相邻路口的会对本路口造成影响的信息
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
    return paths[:3]


def find_next(current_edge, all_path):
    paths = all_path
    # paths = all_path[:20]
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
