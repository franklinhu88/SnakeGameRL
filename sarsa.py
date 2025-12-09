import os
import numpy as np
import random
from snake_env import SnakeEnv, state_tuple_to_int

def epsilon_greedy_action(Q, state_idx, epsilon):
    """Same exploration strategy as Q-learning."""
    if random.random() < epsilon:
        return random.randrange(Q.shape[1])
    return int(np.argmax(Q[state_idx]))

def train(
    env: SnakeEnv,
    n_episodes=20000,
    max_steps_per_episode=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    seed=None,
    save_path="sarsa_table.npy",
    log_every=500
):
    """
    Train an on-policy SARSA agent on SnakeEnv.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_states = 2 ** 11
    n_actions = 3
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        s_idx = state_tuple_to_int(state)

        # SARSA initial action
        action = epsilon_greedy_action(Q, s_idx, epsilon)

        total_reward = 0.0

        for step in range(max_steps_per_episode):
            next_state, reward, done, info = env.step(action)
            s2_idx = state_tuple_to_int(next_state)

            # SARSA chooses next action with epsilon-greedy
            next_action = epsilon_greedy_action(Q, s2_idx, epsilon)

            # SARSA TD target
            td_target = reward + gamma * Q[s2_idx, next_action] * (0 if done else 1)

            # Update
            td_error = td_target - Q[s_idx, action]
            Q[s_idx, action] += alpha * td_error

            total_reward += reward

            s_idx = s2_idx
            action = next_action

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if ep % log_every == 0:
            avg_recent = np.mean(episode_rewards[-log_every:])
            print(f"SARSA Episode {ep}/{n_episodes} | avg_reward={avg_recent:.2f} | epsilon={epsilon:.4f}")

    # ----------------------------
    # SAVE Q-table
    # ----------------------------
    np.save(save_path, Q)
    print(f"SARSA training complete. Saved table to {save_path}")

    # ----------------------------
    # SAVE TRAINING CURVE
    # ----------------------------
    np.save("sarsa_rewards.npy", episode_rewards)
    print("Saved SARSA reward curve to sarsa_rewards.npy")

    return Q, episode_rewards


def evaluate(env: SnakeEnv, Q, n_episodes=10, render=True, max_steps_per_episode=2000):
    scores = []
    for ep in range(n_episodes):
        state = env.reset()
        s_idx = state_tuple_to_int(state)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = int(np.argmax(Q[s_idx]))
            state, reward, done, info = env.step(action)
            s_idx = state_tuple_to_int(state)
            total_reward += reward
            steps += 1

            if render:
                env.render()

        scores.append(info["score"])
        print(f"SARSA Eval Episode {ep+1}: score={scores[-1]}, steps={steps}, total_reward={total_reward:.2f}")

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--save", type=str, default="sarsa_table.npy")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render_eval", action="store_true")

    args = parser.parse_args()

    train_env = SnakeEnv(width=200, height=200, block_size=20, render_mode=False, seed=42)
    render_env = SnakeEnv(width=640, height=480, block_size=20, render_mode=True, speed=10)

    if args.train:
        Q, rewards = train(train_env, n_episodes=args.episodes, save_path=args.save)
        # (Already saved above, but keep this for safety)
        np.save("sarsa_rewards.npy", rewards)

    if args.eval:
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"{args.save} not found. Train SARSA first.")
        Q_loaded = np.load(args.save)
        evaluate(render_env, Q_loaded, n_episodes=5, render=args.render_eval)