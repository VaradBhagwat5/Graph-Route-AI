# Graph Route AI

The RL Routing Simulator is a reinforcement learning project designed to train an agent to find near-optimal paths in a graph environment. It compares the learned paths from a **PPO (Proximal Policy Optimization)** agent against Dijkstra’s algorithm to evaluate performance.

**➡️ Live Demo: [Graph Route AI on Streamlit Community Cloud](https://graph-route-ai.streamlit.app/)**

## Key Features

  - Custom routing environment with a variable number of nodes.
  - PPO-based RL agent for path planning.
  - Evaluation and visualization of RL paths vs. Dijkstra's optimal paths.
  - Streamlit web interface for interactive evaluation.

-----

## Table of Contents

  - [Repository Structure](#repository-structure)
  - [Requirements](#requirements)
  - [Usage](#usage)
      - [Training the RL Agent](#training-the-rl-agent)
      - [Evaluating the RL Agent](#evaluating-the-rl-agent)
      - [Running the Streamlit App](#running-the-streamlit-app)
  - [Notes](#notes)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

-----

## Repository Structure

```
rl-routing-project/
│
├── routing/          # Custom routing environment code
├── create_graph.py   # Dijkstra algorithm and graph plotting
├── main.py           # RL model training and evaluation script
├── app.py            # Streamlit app for evaluation and visualization
├── model.zip         # Saved PPO model (after training)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

-----

## Requirements

To get started, clone the repository and install the required dependencies.

1.  **Prerequisites**:

      * Python 3.8+

2.  **Installation**:
    Install all dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    > **Note**: For GPU support, ensure you install the correct PyTorch version with CUDA by following the instructions on the [official PyTorch site](https://pytorch.org/get-started/locally/).

-----

## Usage

### Training the RL Agent

The PPO agent can be trained using the `main.py` script.

**Default Training Command:**

```bash
python main.py
```

**Custom Training with CLI Arguments:**
You can customize the training process using the following optional arguments:

```bash
python main.py --nodes 20 --epochs 10 --timesteps 200000 --source 0 --destination 5 --render --save_path model.zip
```

**Parameters:**

| Argument        | Description                                       | Default     |
|-----------------|---------------------------------------------------|-------------|
| `--nodes`       | Number of nodes in the graph                      | `10`        |
| `--epochs`      | Number of training epochs                         | `10`        |
| `--timesteps`   | Timesteps per epoch                               | `100000`    |
| `--source`      | Source node for evaluation                        | `6`         |
| `--destination` | Destination node for evaluation                   | `5`         |
| `--render`      | Render the environment and show the evaluation graph | `False`     |
| `--save_path`   | File path to save the trained model               | `model.zip` |

### Evaluating the RL Agent

After training, the agent is automatically evaluated for one episode. The evaluation produces a dual-graph visualization showing:

  - **Dijkstra Path**: The optimal path calculated using Dijkstra’s algorithm.
  - **RL Agent Path**: The path chosen by the trained PPO agent.
  - **Path Costs**: A comparison of the total cost for both paths.

The graph is automatically displayed if the `--render` flag is used during training.

### Running the Streamlit App

The Streamlit app provides an interactive interface to evaluate the trained RL agent and visualize paths in real-time.

**Run the App:**

```bash
streamlit run app.py
```

> **Note**: Ensure a trained `model.zip` file exists in the project's root directory before running the Streamlit app.

**Streamlit App Link**: [link](https://graph-route-ai.streamlit.app/)

-----

## Notes

  - The environment can be customized by adjusting the number of nodes or the source/destination nodes via the CLI arguments in `main.py`.

-----

## License

This repository is released under the [MIT License](https://opensource.org/licenses/MIT).

-----

## Acknowledgements

This project utilizes the following open-source libraries:

- [Gymnasium](https://gymnasium.farama.org/) for the environment API.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms.
- [NetworkX](https://networkx.org/) for graph handling and visualization.
- [Matplotlib](https://matplotlib.org/) for plotting.