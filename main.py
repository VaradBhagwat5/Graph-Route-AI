from routing import Routing
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from create_graph import dijkstra_route, draw_subplots_graph
import matplotlib.pyplot as plt

def train_model(number_of_nodes=10, epochs=10, timesteps=100_000, 
                source=6, destination=5, render=False, save_path="model.zip"):
    """
    Train PPO on the Routing environment, evaluate one episode, and plot results.
    
    Args:
        number_of_nodes (int): Number of nodes in the graph
        epochs (int): Number of training epochs
        timesteps (int): Timesteps per epoch
        source (int): Source node for evaluation
        destination (int): Destination node for evaluation
        render (bool): Whether to render the environment
        save_path (str): File path to save the trained model
    Returns:
        model: Trained PPO model
        fig: Matplotlib figure of evaluation graph
    """
    # Create environment
    env = Routing(number_of_nodes=number_of_nodes, render_mode="human" if render else None)
    check_env(env)

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train
    model.learn(total_timesteps=epochs*timesteps)
    
    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Evaluation on one episode
    obs, _ = env.reset(options={"source": source, "destination": destination})
    G, pos = env.G, env.pos
    dijkstra_path, dijkstra_cost, weight_labels = dijkstra_route(G, source, destination)
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    
    # Plot evaluation results
    fig = draw_subplots_graph(
        G, pos, dijkstra_path, env.path_taken,
        source, destination, info['path_weight'], dijkstra_cost, weight_labels
    )
    
    # Show the figure if render=True
    if render:
        plt.show()
    
    return model, fig

# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on Routing environment")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per epoch")
    parser.add_argument("--source", type=int, default=6, help="Source node for evaluation")
    parser.add_argument("--destination", type=int, default=5, help="Destination node for evaluation")
    parser.add_argument("--render", action="store_true", help="Render environment and plot")
    parser.add_argument("--save_path", type=str, default="model.zip", help="Path to save trained model")
    
    args = parser.parse_args()

    train_model(
        number_of_nodes=args.nodes,
        epochs=args.epochs,
        timesteps=args.timesteps,
        source=args.source,
        destination=args.destination,
        render=args.render,
        save_path=args.save_path
    )
