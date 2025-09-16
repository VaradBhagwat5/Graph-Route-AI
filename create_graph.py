import networkx as nx
import random
from weight import calculate_latency, compute_weight, path_total_weight
import matplotlib.pyplot as plt

def create_edges(nodes_number):
    """
    Creates random number of connections (1, n-1) to random nodes
    """
    connections = []
    for i in range(1, nodes_number+1):
        number_of_connections = random.randint(1, nodes_number-1)   # possibility of making n-1 connections
        for j in range(number_of_connections):
            nodes = [i for i in range(1, nodes_number+1)]
            nodes.remove(i)   # should not make connections with itself, so removed the i'th node
            connections.append((i, random.choice(nodes)))  # selects random nodes to make connections with
    return connections

def create_graph(nodes_number):
    """
    Creates the networkx graph, adds nodes, adds edges, adds weights between the edges
    Latency is determined via distance between two respective nodes.
    Bandwidth and Congestion are created randomly between two nodes.
    """
    G = nx.Graph()   # initiates the graph

    nodes = [i for i in range(1, nodes_number+1)]   # Generate n number of nodes
    G.add_nodes_from(nodes)   # add nodes to graph

    pos = nx.random_layout(G)   # Generates random positions for nodes, returns a dict with nodes value as keys and x, y positions as value

    edges = create_edges(nodes_number)

    for u, v in edges:   # u and v are nodes
        latency = calculate_latency(pos, u, v)   # in ms
        bandwidth = random.randint(10, 1000)   # mbps
        congestion = round(random.uniform(0.1, 1.0), 2)   # conjestion between 0 and 1, 1 being 100% congested
        alpha = 100
        G.add_edge(u, v, latency=latency, bandwidth=bandwidth, congestion=congestion)

    return G, pos

def dijkstra_route(G, source, destination):
    """
    Returns the route dijkstra takes for a given graph, the optimal way along with the total cost it takes to reach destination from source
    Also returns the weights between two nodes (on edges) using special compute function.
    """
    # create a dict of weights by recomputing for each edge
    weight_labels = {
        (u, v): round(compute_weight(u, v, d), 2) 
        for u, v, d in G.edges(data=True)
    }
    path = nx.shortest_path(G, source, destination, weight=compute_weight)   # calculate the shortest path
    total_cost = path_total_weight(G, path, compute_weight)   # the total cost of path taken by dijkstra
    return path, total_cost, weight_labels

def draw_subplots_graph(G, pos, dijkstra_path, agent_path, source, destination, cost_agent, cost_dijkstra, weight_labels):
    """
    Creates a figure plotting both graphs from dijkstra and RL learnt
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("Dijkstra vs RL Agent")
    
    # Subplot 1: Dijkstra's Path
    ax1.set_title(f'Dijkstra\nCost: {cost_dijkstra:.5f}\nDijkstra Path: {dijkstra_path}')
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
    dijkstra_edges = list(zip(dijkstra_path, dijkstra_path[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=dijkstra_edges, edge_color='green', width=2)
    nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[source], node_color='lightgreen', node_size=1000)
    nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[destination], node_color='salmon', node_size=1000)
    nx.draw_networkx_edge_labels(G, pos, ax=ax1, edge_labels=weight_labels)

    # Subplot 2: Agent's Path
    ax2.set_title(f'RL Agent\nCost: {cost_agent:.5f}\nRL Path: {agent_path}')
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
    agent_edges = list(zip(agent_path, agent_path[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=agent_edges, edge_color='red', width=2)
    nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=[source], node_color='lightgreen', node_size=1000)
    nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=[destination], node_color='salmon', node_size=1000)
    nx.draw_networkx_edge_labels(G, pos, ax=ax2, edge_labels=weight_labels)

    plt.tight_layout()
    return fig