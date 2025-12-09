# linear_q.py
import numpy as np
import random
from snake_env import SnakeEnv, state_tuple_to_int

"""
Linear Function Approximation Q-learning
----------------------------------------
Q(s, a) = w[a] · x(s)
Where:
- w[a] is a weight vector for action a
- x(s) is an 11-dim feature vector (binary)
"""

def featurize(state):
    return np.array(state, dtype=np.float32)


def epsilon_greedy(w, x, epsilon):
    if random.random() < epsilon:
        return random.randrange(w.shape[0])
    return int(np.argmax(w @ x))


def train(
    env: SnakeEnv,
    n_episodes=20000,
    max_steps_per_episode=1000,
    alpha=0.01,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    seed=None,
    save_path="linear_q_weights.npy",
    log_every=500
):
    """
    Linear function approximation Q-learning.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_features = 11
    n_actions = 3

    # (3 x 11) weight matrix
    w = np.zeros((n_actions, n_features), dtype=np.float32)

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        x = featurize(state)
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            # Choose action
            a = epsilon_greedy(w, x, epsilon)

            # Step env
            next_state, reward, done, info = env.step(a)
            x2 = featurize(next_state)

            # TD target
            q_next = w @ x2
            best_next = np.max(q_next)
            td_target = reward + gamma * best_next * (0 if done else 1)

            # TD error
            q_sa = w[a] @ x
            td_error = td_target - q_sa

            # Update weights
            w[a] += alpha * td_error * x

            total_reward += reward
            x = x2

            if done:
                break

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if ep % log_every == 0:
            avg = np.mean(episode_rewards[-log_every:])
            print(f"[Linear-Q] Episode {ep}/{n_episodes} | avg_reward={avg:.2f} | epsilon={epsilon:.3f}")

    # -------------------------------
    # SAVE LEARNED WEIGHTS
    # -------------------------------
    np.save(save_path, w)
    print(f"Saved Linear-Q weights to {save_path}")

    # -------------------------------
    # SAVE TRAINING CURVE
    # -------------------------------
    np.save("linear_q_rewards.npy", episode_rewards)
    print("Saved linear_q_rewards.npy")

    return w, episode_rewards


def evaluate(env: SnakeEnv, w, n_episodes=10, render=True):
    scores = []

    for ep in range(n_episodes):
        state = env.reset()
        x = featurize(state)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 2000:
            q_vals = w @ x
            action = int(np.argmax(q_vals))

            next_state, reward, done, info = env.step(action)
            x = featurize(next_state)

            total_reward += reward
            steps += 1
            if render:
                env.render()

        scores.append(info["score"])
        print(f"Linear-Q Eval Ep {ep+1}: score={scores[-1]}, steps={steps}, total_reward={total_reward:.2f}")

    return scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--save", type=str, default="linear_q_weights.npy")
    args = parser.parse_args()

    env = SnakeEnv(width=200, height=200, block_size=20, render_mode=False, seed=42)

    print("Starting Linear-Q training...")
    w, rewards = train(env, n_episodes=args.episodes, save_path=args.save)

    np.save("linear_q_rewards.npy", rewards)
    print("Saved linear_q_rewards.npy")