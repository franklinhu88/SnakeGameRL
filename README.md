# Snake Game with Q-Learning

A Snake game implementation with a reinforcement learning agent that learns to play using Q-Learning. CS4260 Final Project.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SnakeGameRL.git
cd SnakeGameRL

# Create virtual environment (Python 3.12)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Play the Game Manually
```bash
python snake_app.py
```
Use arrow keys to control the snake.

### Train the Q-Learning Agent
```bash
python q_learning.py --train --episodes 20000 --save q_table.npy
```

Parameters:
- `--train`: Enable training mode
- `--episodes [N]`: Number of training episodes
- `--save [filename]`: Save the trained Q-table
- `--load [filename]`: Load existing Q-table
- `--render`: Visualize training (slower)

### Run Trained Agent
```bash
python q_learning.py --load q_table.npy --render
```

## Speed Controls (Visualization Mode)

Control playback speed when watching the agent:

| Key | Action |
|-----|--------|
| `←` | Slow down |
| `→` | Speed up |
| `1` | Normal speed |
| `2` | Fast |
| `3` | Faster |
| `4` | Fastest |

## Q-Learning Implementation

The agent learns through trial and error using the Q-Learning algorithm with:

- **State Features**: Food direction, danger detection, snake direction
- **Actions**: Up, Down, Left, Right
- **Rewards**: +10 for food, -10 for collision, ±1 for distance to food
- **Q-Update**: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`

### Hyperparameters
```python
LEARNING_RATE = 0.1       # Alpha
DISCOUNT_FACTOR = 0.95    # Gamma  
EPSILON_START = 1.0       # Initial exploration
EPSILON_DECAY = 0.995     # Exploration decay
EPSILON_MIN = 0.01        # Minimum exploration
```

## Project Structure
```
SnakeGameRL/
├── snake_app.py          # Snake game implementation
├── q_learning.py         # Q-Learning agent
├── requirements.txt      # Dependencies
├── q_table.npy          # Saved model
└── README.md            
```

## Authors

- Franklin Hu
- Jonathan Kim
