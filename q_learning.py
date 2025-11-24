# q_learning.py
import os
import numpy as np
import random
from snake_env import SnakeEnv, state_tuple_to_int

def epsilon_greedy_action(Q, state_idx, epsilon):
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
    save_path="q_table.npy",
    log_every=500
):
    """
    Train a tabular Q-learning agent on SnakeEnv.
    - env: SnakeEnv instance (use render_mode=False for training)
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_states = 2 ** 11  # 11 binary features -> 2048
    n_actions = 3       # straight, right, left
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()  # returns state tuple
        s_idx = state_tuple_to_int(state)
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            a = epsilon_greedy_action(Q, s_idx, epsilon)
            next_state, reward, done, info = env.step(a)
            s2_idx = state_tuple_to_int(next_state)

            # Q-learning update
            best_next = np.max(Q[s2_idx])
            td_target = reward + gamma * best_next * (0 if done else 1)
            td_error = td_target - Q[s_idx, a]
            Q[s_idx, a] += alpha * td_error

            total_reward += reward
            s_idx = s2_idx

            if done:
                break

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if ep % log_every == 0 or ep == 1:
            avg_recent = np.mean(episode_rewards[-log_every:]) if len(episode_rewards) >= log_every else np.mean(episode_rewards)
            print(f"Episode {ep}/{n_episodes} | avg_reward_last_{min(ep,log_every)}={avg_recent:.3f} | epsilon={epsilon:.4f}")

    # save Q-table
    np.save(save_path, Q)
    print(f"Training finished. Saved Q-table to {save_path}")
    return Q, episode_rewards

def evaluate(env: SnakeEnv, Q, n_episodes=10, render=True, max_steps_per_episode=2000, seed=None):
    """
    Evaluate policy greedy w.r.t. Q.
    Returns list of scores.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    scores = []
    for ep in range(n_episodes):
        state = env.reset()
        s_idx = state_tuple_to_int(state)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            # greedy policy
            a = int(np.argmax(Q[s_idx]))
            state, reward, done, info = env.step(a)
            s_idx = state_tuple_to_int(state)
            total_reward += reward
            steps += 1
            if render:
                env.render()

        scores.append(info.get('score', 0))
        print(f"Eval Episode {ep+1}: score={scores[-1]}, steps={steps}, total_reward={total_reward:.2f}")
    return scores

if __name__ == "__main__":
    # Minimal CLI usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run training")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--save", type=str, default="q_table.npy")
    parser.add_argument("--eval", action="store_true", help="run evaluation using saved q_table.npy")
    parser.add_argument("--render_eval", action="store_true", help="render during eval")
    args = parser.parse_args()

    # Use a small env for faster training if you wish:
    train_env = SnakeEnv(width=200, height=200, block_size=20, render_mode=False, seed=42)
    render_env = SnakeEnv(width=640, height=480, block_size=20, render_mode=True, speed=10)

    if args.train:
        Q, rewards = train(train_env, n_episodes=args.episodes, save_path=args.save)
    if args.eval:
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"{args.save} not found. Train first.")
        Q_loaded = np.load(args.save)
        evaluate(render_env, Q_loaded, n_episodes=5, render=args.render_eval)
