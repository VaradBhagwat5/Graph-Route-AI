import math

def calculate_latency(pos, u, v):
    """
    Calculate the latency between two nodes using the distance between them.
    """
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # using euclidian distance
    latency = int(distance * 100)  # Multiplies by 100 for a reasonable value
    return latency

def compute_weight(current, next, data):
    """
    Calculate the weight between two nodes by combining latency, bandwidth, congestion. 
    """
    # Handle both Graph and MultiGraph edge data
    if 0 in data:  # multigraph-style dict
        data = data[0]
    # big brain by chatgpt
    latency = data['latency']
    bandwidth = data['bandwidth']
    congestion = data['congestion']
    alpha = 100
    return (latency * congestion + alpha) / bandwidth

def path_total_weight(G, path, weight_fn):
    """
    Calculate the total path cost by RL agent
    """
    total = 0
    for u, v in zip(path, path[1:]):
        data = G[u][v]  # this is the edge attribute dict
        # handle MultiGraph case
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        total += weight_fn(u, v, data)
    return total