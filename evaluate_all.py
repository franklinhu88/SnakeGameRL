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
def make_env(seed: int | None = 42) -> SnakeEnv:
    """Environment for FAST evaluation (same size as Q-learning training)."""
    env = SnakeEnv(
        width=200,
        height=200,
        block_size=20,
        render_mode=False,
        seed=seed
    )
    return env


# ------------------------------
# Q-Learning agent wrapper
# ------------------------------
def q_learning_agent_factory(Q: np.ndarray):
    def agent(state, env):
        idx = state_tuple_to_int(state)
        return int(np.argmax(Q[idx]))
    return agent


# ------------------------------
# SARSA agent wrapper
# ------------------------------
def sarsa_agent_factory(Q: np.ndarray):
    def agent(state, env):
        idx = state_tuple_to_int(state)
        return int(np.argmax(Q[idx]))  # greedy eval
    return agent


# ------------------------------
# Core evaluation function
# ------------------------------
def evaluate_agent(
    env: SnakeEnv,
    agent_fn,
    n_episodes: int = 50,
    max_steps_per_episode: int = 1000
) -> np.ndarray:

    scores = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        info = {}

        while not done and steps < max_steps_per_episode:
            action = agent_fn(state, env)
            state, reward, done, info = env.step(action)
            steps += 1

        scores.append(info.get("score", 0))

    return np.asarray(scores, dtype=np.float32)


# ------------------------------
# Pretty results table
# ------------------------------
def print_results_table(agent_scores: dict[str, np.ndarray]) -> None:
    print("\nAgent Performance Summary:\n")
    header = "{:<15} {:>10} {:>10} {:>10}".format(
        "Agent", "Avg", "Max", "Std"
    )
    print(header)
    print("-" * len(header))

    for name, scores in agent_scores.items():
        avg = float(scores.mean())
        mx = float(scores.max())
        std = float(scores.std())
        line = "{:<15} {:>10.2f} {:>10.0f} {:>10.2f}".format(
            name, avg, mx, std
        )
        print(line)


# ------------------------------
# Visualization
# ------------------------------
def plot_average_scores(agent_scores: dict[str, np.ndarray], out="agent_avg_scores.png"):
    names = list(agent_scores.keys())
    avgs = [float(v.mean()) for v in agent_scores.values()]
    stds = [float(v.std()) for v in agent_scores.values()]

    plt.figure()
    x = np.arange(len(names))
    plt.bar(x, avgs, yerr=stds, capsize=5)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Average Score")
    plt.title("Average Score per Agent")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_score_boxplots(agent_scores: dict[str, np.ndarray], out="agent_score_boxplots.png"):
    names = list(agent_scores.keys())
    data = [v for v in agent_scores.values()]

    plt.figure()
    plt.boxplot(data, labels=names, showmeans=True)
    plt.ylabel("Score")
    plt.title("Score Distribution per Agent")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_training_rewards(rewards_path="q_rewards.npy", out="q_training_rewards.png"):
    if not os.path.exists(rewards_path):
        print(f"No {rewards_path}. Skipping training reward plot.")
        return

    rewards = np.load(rewards_path)

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Reward Curve")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ------------------------------
# Main Evaluation Logic
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per agent for evaluation.")
    parser.add_argument("--q_path", type=str, default="q_table.npy",
                        help="Path to trained Q-learning table.")
    parser.add_argument("--sarsa_path", type=str, default="sarsa_table.npy",
                        help="Path to trained SARSA table (optional).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(seed=args.seed)

    agent_scores = {}

    # ------------------------------
    # 1. Baselines
    # ------------------------------
    for name, agent_fn in AGENT_BASELINES.items():
        print(f"Evaluating baseline: {name} ...")
        scores = evaluate_agent(env, agent_fn, n_episodes=args.episodes)
        agent_scores[name] = scores

    # ------------------------------
    # 2. Q-learning agent
    # ------------------------------
    if os.path.exists(args.q_path):
        print("\nEvaluating Q-learning agent...")
        Q = np.load(args.q_path)
        q_agent = q_learning_agent_factory(Q)
        agent_scores["q_learning"] = evaluate_agent(env, q_agent, n_episodes=args.episodes)
    else:
        print("No Q-learning table found; skipping Q-learning evaluation.")

    # ------------------------------
    # 3. SARSA agent
    # ------------------------------
    if os.path.exists(args.sarsa_path):
        print("\nEvaluating SARSA agent...")
        S = np.load(args.sarsa_path)
        sarsa_agent = sarsa_agent_factory(S)
        agent_scores["sarsa"] = evaluate_agent(env, sarsa_agent, n_episodes=args.episodes)
    else:
        print("No SARSA table found; skipping SARSA evaluation.")

    print_results_table(agent_scores)
    plot_average_scores(agent_scores)
    plot_score_boxplots(agent_scores)
    plot_training_rewards()


if __name__ == "__main__":
    main()