# SnakeGameRL
Snake Game Agent for CS4260 Final Project

To run, pull the repo, then create a virtual environment (Project works in Python 3.12)

Then run 
```bash
pip install -r requirements.txt
```

If you want to play the game, just run `python snake_app.py`

To train an agent, run `python q_learning.py --train --episodes 20000 --save q_table.npy` to train 20,000 episodes of the agent
