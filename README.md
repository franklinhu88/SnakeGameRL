# SnakeGameRL

A Snake game implementation with multiple reinforcement learning agents, including **Q-Learning**, **SARSA**, **Linear-Q**, and several baseline policies.

Final Project for **CS4260: Artificial Intelligence**.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SnakeGameRL.git
cd SnakeGameRL

# Create virtual environment (Python 3.12)
python -m venv venv
source venv/bin/activate              # macOS/Linux
venv\Scripts\activate                 # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Play the Game Manually
```bash
python snake_app.py
```

Use **arrow keys** to control the snake.

---

## Train Reinforcement Learning Agents

### Q-Learning
```bash
python q_learning.py --train --episodes 20000 --save q_table.npy
```

### SARSA
```bash
python sarsa.py --train --episodes 20000 --save sarsa_table.npy
```

### Linear-Q Function Approximation
```bash
python linear_q.py --episodes 20000 --save linear_q_weights.npy
```

Each training script automatically saves a reward curve:

- `q_rewards.npy`
- `sarsa_rewards.npy`
- `linear_q_rewards.npy`

---

## Evaluate All Agents (Baselines + RL)

Runs:

- Random Policy
- Safe-Survival Policy
- Greedy-Food Policy
- Untrained Q-table
- Q-Learning
- SARSA
- Linear-Q

Generates performance plots and tables.
```bash
python evaluate_all.py --episodes 50
```

Outputs:

- `agent_avg_scores.png`
- `agent_score_boxplots.png`
- `q_training_rewards.png`
- `sarsa_training_rewards.png`
- `linear_q_training_rewards.png`
- `combined_training_curves.png`

---

## Run a Trained Agent (Playback Mode)

### Q-Learning
```bash
python q_learning.py --eval --render_eval
```

### SARSA
```bash
python sarsa.py --eval --render_eval
```

### Linear-Q
```bash
python linear_q.py --eval
```

---

## Speed Controls (Visualization Mode)

| Key | Action |
|-----|--------|
| ← | Slow down |
| → | Speed up |
| 1 | Normal speed |
| 2 | Fast |
| 3 | Faster |
| 4 | Fastest |

---

## Q-Learning Overview

The agent uses an **11-dimensional binary feature vector** describing:

- Danger straight / left / right
- Current movement direction (one-hot)
- Relative food direction

**Actions:**

- Move Forward
- Turn Left
- Turn Right

**Rewards:**

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Die | −10 |
| Each step | −0.01 |

**Q-update rule:**
```bash
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

---

## Project Structure
```graphql
SnakeGameRL/
├── snake_app.py              # Game engine
├── snake_env.py              # RL environment wrapper
├── baselines.py              # Baseline agents
├── q_learning.py             # Q-learning agent
├── sarsa.py                  # SARSA agent
├── linear_q.py               # Linear Q-function agent
├── evaluate_all.py           # Evaluation + plotting
└── README.md
```

---

## Authors

- Jonathan Kim
- Franklin Hu
