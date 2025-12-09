import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

from snake_env import SnakeEnv, state_tuple_to_int
from baselines import AGENT_BASELINES


# ------------------------------
# Utility: Create environment
# ------------------------------
def make_env(seed=42):
    return SnakeEnv(
        width=200,
        height=200,
        block_size=20,
        render_mode=False,
        seed=seed
    )


# ------------------------------
# Agent wrappers
# ------------------------------
def q_learning_agent_factory(Q):
    def agent(state, env):
        return int(np.argmax(Q[state_tuple_to_int(state)]))
    return agent


def sarsa_agent_factory(Q):
    def agent(state, env):
        return int(np.argmax(Q[state_tuple_to_int(state)]))
    return agent


# ------------------------------
# Core evaluation loop
# ------------------------------
def evaluate_agent(env, agent_fn, n_episodes=50, max_steps_per_episode=1000):
    scores = []

    for _ in range(n_episodes):
        state = env.reset()
        done, steps = False, 0
        info = {}

        while not done and steps < max_steps_per_episode:
            action = agent_fn(state, env)
            state, reward, done, info = env.step(action)
            steps += 1

        scores.append(info.get("score", 0))

    return np.asarray(scores, dtype=np.float32)


# ------------------------------
# Results Table
# ------------------------------
def print_results_table(agent_scores):
    print("\nAgent Performance Summary:\n")
    header = "{:<15} {:>10} {:>10} {:>10}".format(
        "Agent", "Avg", "Max", "Std"
    )
    print(header)
    print("-" * len(header))

    for name, scores in agent_scores.items():
        print("{:<15} {:>10.2f} {:>10.0f} {:>10.2f}".format(
            name, scores.mean(), scores.max(), scores.std()
        ))


# ------------------------------
# Visualization Helpers
# ------------------------------
def plot_average_scores(agent_scores, out="agent_avg_scores.png"):
    names = list(agent_scores.keys())
    avgs = [v.mean() for v in agent_scores.values()]
    stds = [v.std() for v in agent_scores.values()]

    plt.figure()
    x = np.arange(len(names))
    plt.bar(x, avgs, yerr=stds, capsize=5)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Average Score")
    plt.title("Average Score per Agent")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_score_boxplots(agent_scores, out="agent_score_boxplots.png"):
    plt.figure()
    plt.boxplot(agent_scores.values(), labels=agent_scores.keys(), showmeans=True)
    plt.ylabel("Score")
    plt.title("Score Distribution per Agent")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ------------------------------
# Training reward curve (generalized)
# ------------------------------
def plot_training_rewards(path, out, title):
    if not os.path.exists(path):
        print(f"Missing: {path}, skipping curve.")
        return

    rewards = np.load(path)

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"Saved {out}")


# ------------------------------
# Combined multi-line overlay curve
# ------------------------------
def plot_combined_training_curves(curve_dict, out="combined_training_curves.png"):
    plt.figure()

    for name, file in curve_dict.items():
        if os.path.exists(file):
            rewards = np.load(file)
            plt.plot(rewards, label=name)
        else:
            print(f"Skipping missing curve: {file}")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curves (Comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"Saved combined training curves to {out}")


# ------------------------------
# Main Evaluation Logic
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--q_path", type=str, default="q_table.npy")
    parser.add_argument("--sarsa_path", type=str, default="sarsa_table.npy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(seed=args.seed)
    agent_scores = {}

    # Baselines
    for name, agent_fn in AGENT_BASELINES.items():
        print(f"Evaluating baseline: {name} ...")
        agent_scores[name] = evaluate_agent(env, agent_fn, args.episodes)

    # Q-learning
    if os.path.exists(args.q_path):
        print("\nEvaluating Q-learning agent...")
        Q = np.load(args.q_path)
        agent_scores["q_learning"] = evaluate_agent(env, q_learning_agent_factory(Q), args.episodes)

    # SARSA
    if os.path.exists(args.sarsa_path):
        print("\nEvaluating SARSA agent...")
        S = np.load(args.sarsa_path)
        agent_scores["sarsa"] = evaluate_agent(env, sarsa_agent_factory(S), args.episodes)

    # Linear-Q
    if os.path.exists("linear_q_weights.npy"):
        print("\nEvaluating Linear-Q agent...")
        W = np.load("linear_q_weights.npy")

        def linear_q_agent(state, env):
            x = np.array(state, dtype=np.float32)
            return int(np.argmax(W @ x))

        agent_scores["linear_q"] = evaluate_agent(env, linear_q_agent, args.episodes)

    # Print table + plots
    print_results_table(agent_scores)
    plot_average_scores(agent_scores)
    plot_score_boxplots(agent_scores)

    # Individual training curves
    plot_training_rewards("q_rewards.npy", "q_training_rewards.png", "Q-learning Training Curve")
    plot_training_rewards("sarsa_rewards.npy", "sarsa_training_rewards.png", "SARSA Training Curve")
    plot_training_rewards("linear_q_rewards.npy", "linear_q_training_rewards.png", "Linear-Q Training Curve")

    # Combined training curve overlay
    plot_combined_training_curves({
        "Q-learning": "q_rewards.npy",
        "SARSA": "sarsa_rewards.npy",
        "Linear-Q": "linear_q_rewards.npy"
    })


if __name__ == "__main__":
    main()