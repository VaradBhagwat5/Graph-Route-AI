import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
np.bool8 = bool   # backward compatibility fix for numpy
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
from create_graph import create_graph


class Routing(gym.Env):
    """
    Custom Gym environment for simulating routing in a network graph.
    Agent learns to navigate from source to destination node
    with latency, bandwidth, and congestion considered as edge weights.
    """
    def __init__(self, number_of_nodes=10, render_mode=None):
        super(Routing, self).__init__()
        
        # ---- Graph-related variables ----
        self.G = None                  # Graph object (NetworkX)
        self.source = None             # Source node
        self.destination = None        # Destination node
        self.current = None            # Current node agent is at
        self.number_of_nodes = number_of_nodes
        self.pos = None                # Node positions for plotting

        # ---- Environment limits ----
        self.max_steps = 50            # Max steps allowed per episode
        self.invalid_action_penalty = -3
        self.finished_reward = 10      # Reward when destination is reached
        self.maximum_edge_cost = 0     # Track highest cost edge (used for normalization)

        # ---- Gym spaces ----
        # Action: choose next node (Discrete)
        self.action_space = Discrete(self.number_of_nodes)
        
        # Observation: concatenated features (one-hots + costs)
        observation_length = 4 * self.number_of_nodes
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_length,),
            dtype=np.float32
        )

        # ---- Episode tracking ----
        self.steps = 0
        self.total_cost = 0
        self.done = False
        self.path_taken = []           # Track the sequence of visited nodes
        self.path_weight = 0
        self.neighbours = {}           # Dict of node → neighbours
        self.render_mode = render_mode

    def step(self, action):
        """
        Take one step in the environment:
        - Check if action is valid
        - Move agent to next node if valid, else penalize
        - Compute reward based on weighted cost
        - End episode if destination is reached or max steps exceeded
        """
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False

        # If already at destination → terminate
        if self.current == self.destination:
            terminated = True

        # ---- Valid move ----
        if action in self.neighbours[self.current]:
            edge_data = self.G.get_edge_data(self.current, action)
            step_cost = self.compute_weight(0, 0, edge_data)
            self.total_cost += step_cost

            # Distance heuristic (not directly used but can be useful later)
            old_distance = self.calculate_distance(self.current, self.destination)
            new_distance = self.calculate_distance(action, self.destination)

            # Reward = negative cost (normalized)
            reward = -(self.total_cost / self.maximum_edge_cost)

            # Update current state
            self.current = action
            self.path_taken.append(action)

        # ---- Invalid move ----
        else:
            reward = self.invalid_action_penalty

        # Penalize cycles (discourage revisiting)
        if action in self.path_taken:
            reward += self.invalid_action_penalty

        # ---- Step bookkeeping ----
        self.steps += 1

        # ---- Check termination ----
        if self.current == self.destination:
            total_cost = self.path_total_weight(self.G, self.path_taken, self.compute_weight)
            reward += self.finished_reward - total_cost
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        # ---- Return Gym API outputs ----
        return (
            self.get_observation_space(),
            float(reward),
            terminated,
            truncated,
            {"path_weight": self.path_total_weight(self.G, self.path_taken, self.compute_weight)}
        )

    def render(self):
        """
        Render the current graph and path using Matplotlib + NetworkX.
        Draw nodes, all edges, and highlight path edges in red.
        """
        if self.render_mode == "human":
            plt.close()
            nx.draw(self.G, self.pos, with_labels=True)

            # Draw path edges in red
            if len(self.path_taken) > 1:
                path_edges = list(zip(self.path_taken[:-1], self.path_taken[1:]))
                nx.draw_networkx_edges(
                    self.G, self.pos,
                    edgelist=path_edges,
                    edge_color="red",
                    width=2
                )
                # Add computed weights as edge labels
                weight_labels = {
                    (u, v): round(self.compute_weight(u, v, d), 2) 
                    for u, v, d in self.G.edges(data=True)
                }
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=weight_labels)

            plt.title("PPO's Route", fontsize=16)
            plt.pause(0.5)  # pause for animation effect

    def calculate_distance(self, u, v):
        """
        Euclidean distance between two nodes (based on position).
        Used as a heuristic measure (not directly in reward now).
        """
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def compute_weight(self, current, next, data):
        """
        Custom edge weight function.
        Combines latency, bandwidth, and congestion into a single metric.
        """
        latency = data['latency']
        bandwidth = data['bandwidth']
        congestion = data['congestion']
        alpha = 100  # bias factor to keep weights positive
        return (latency * congestion + alpha) / bandwidth
    
    def path_total_weight(self, G, path, weight_fn):
        """
        Compute total path cost using the custom weight function.
        Works as a replacement for nx.path_weight which only supports string keys.
        """
        total = 0
        for u, v in zip(path, path[1:]):
            data = G[u][v]
            # Handle MultiGraph (edge dict nested)
            if isinstance(data, dict) and 0 in data:
                data = data[0]
            total += weight_fn(u, v, data)
        return total
    
    def get_latency(self, index):
        """
        Construct latency feature vector for current node:
        - Stores computed weight for neighbours
        - Non-neighbours get a large constant (discourages going there)
        """
        vector = np.zeros(self.number_of_nodes)
        for neighbour in self.neighbours[index]:
            data = self.G.get_edge_data(index, neighbour)
            vector[neighbour-1] = self.compute_weight(0, 0, data)
        for i in range(self.number_of_nodes):
            if vector[i] == 0:
                vector[i] = 7   # assign high cost for unreachable
        return vector
    
    def create_onehot(self, index):
        """
        Create one-hot encoding for a node index.
        """
        vector = np.zeros(self.number_of_nodes)
        vector[index-1] = 1
        return vector
    
    def get_neighbour_vector(self, path_taken):
        """
        Create a vector marking which nodes have already been visited.
        Helps agent avoid cycles.
        """
        vector = np.zeros(self.number_of_nodes)
        for i in path_taken:
            vector[i-1] = 1
        return vector
    
    def get_observation_space(self):
        """
        Construct observation vector = concat of:
        - one-hot current node
        - one-hot destination node
        - latency vector for current node
        - visited nodes vector
        """
        current_onehot = self.create_onehot(self.current)
        destination_onehot = self.create_onehot(self.destination)
        latency_onehot = self.get_latency(self.current)
        path_travelled = self.get_neighbour_vector(self.path_taken)
        return np.concatenate(
            [current_onehot, destination_onehot, latency_onehot, path_travelled]
        ).astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment:
        - Generate a fresh graph
        - Choose random or user-specified source & destination
        - Reset counters and states
        """
        super().reset(seed=seed)
        self.G, self.pos = create_graph(self.number_of_nodes)

        # Choose source/destination
        if options and "source" in options and "destination" in options:
            self.source, self.destination = options["source"], options["destination"]
        else:
            self.source, self.destination = random.sample(range(1, self.number_of_nodes+1), 2)

        self.current = self.source
        self.done = False
        self.path_taken = [self.source]

        # Store neighbours for each node
        self.neighbours = {i: list(self.G.neighbors(i)) for i in range(1, self.number_of_nodes+1)}

        # Track maximum edge cost for reward normalization
        for u, v, d in self.G.edges(data=True):
            self.maximum_edge_cost = max(self.maximum_edge_cost, self.compute_weight(u, v, d))

        # Reset counters
        self.steps = 0
        self.total_cost = 0
        return self.get_observation_space(), {}