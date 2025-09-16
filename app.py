import streamlit as st
from routing import Routing
from stable_baselines3 import PPO
from create_graph import dijkstra_route, draw_subplots_graph
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="RL Routing Simulator", layout="wide")
st.title("RL Routing Evaluation (PPO vs Dijkstra)")

# Sidebar inputs
st.sidebar.header("Evaluation Settings")
num_nodes = st.sidebar.number_input("Number of Nodes", min_value=5, max_value=50, value=10)
source = st.sidebar.number_input("Source Node", min_value=0, max_value=num_nodes-1, value=6)
destination = st.sidebar.number_input("Destination Node", min_value=0, max_value=num_nodes-1, value=5)
episodes = st.sidebar.slider("Episodes to run", 1, 10, 1)

eval_button = st.sidebar.button("Run Evaluation")

if eval_button:
    if not os.path.exists("model.zip"):
        st.error("model.zip not found in current directory. Please train and save first.")
    else:
        st.write(f"Evaluating PPO model for {episodes} episode(s)...")

        all_results = []
        for ep in range(episodes):
            env = Routing(number_of_nodes=num_nodes, render_mode="human")
            model = PPO.load("model.zip", env=env)

            obs, _ = env.reset(options={"source": source, "destination": destination})
            G, pos = env.G, env.pos
            dijkstra_path, dijkstra_cost, weight_labels = dijkstra_route(G, source, destination)

            done, info = False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

            all_results.append((ep+1, info['path_weight'], dijkstra_cost))

        # Show results
        st.subheader("Results")
        for ep, ppo_cost, dj_cost in all_results:
            st.write(f"Episode {ep}: PPO Path = {ppo_cost:.2f}, Dijkstra Path = {dj_cost:.2f}")

        avg_ppo = sum([x[1] for x in all_results]) / len(all_results)
        avg_dj = sum([x[2] for x in all_results]) / len(all_results)
        st.write(f"**Average PPO Path Cost:** {avg_ppo:.2f}")
        st.write(f"**Average Dijkstra Path Cost:** {avg_dj:.2f}")

        # Plot last episode’s graph
        fig = draw_subplots_graph(G, pos, dijkstra_path, env.path_taken,
                            source, destination, info['path_weight'], dijkstra_cost, weight_labels)
        st.pyplot(fig)
        st.success("Evaluation complete!")
