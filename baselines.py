import random
import numpy as np
from typing import Callable, Tuple, Any

from snake_env import Direction, state_tuple_to_int

DANGER_STRAIGHT = 0
DANGER_RIGHT = 1
DANGER_LEFT = 2
DIR_RIGHT = 3
DIR_LEFT = 4
DIR_UP = 5
DIR_DOWN = 6
FOOD_LEFT = 7
FOOD_RIGHT = 8
FOOD_UP = 9
FOOD_DOWN = 10

N_STATES = 2 ** 11
N_ACTIONS = 3

Q_ZERO = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

def random_agent(state: Tuple[int, ...], env: Any) -> int:
    """Completely random actions: floor baseline."""
    return random.randint(0, 2)  # 0, 1, or 2


def safe_survival_agent(state: Tuple[int, ...], env: Any) -> int:
    """
    Survival heuristic:
    - If danger straight, try to turn to a safe side.
    - Otherwise, go straight.
    """
    danger_straight = bool(state[DANGER_STRAIGHT])
    danger_right = bool(state[DANGER_RIGHT])
    danger_left = bool(state[DANGER_LEFT])

    if danger_straight:
        if not danger_left:
            return 2  # turn left
        if not danger_right:
            return 1  # turn right
        return 0  # trapped, just go straight
    return 0  # straight


def greedy_food_agent(state: Tuple[int, ...], env: Any) -> int:
    """
    Simple food-seeking heuristic:
    - Pick an absolute direction that moves toward food.
    - Convert that into a relative action (straight/right/left)
      based on the current direction.
    """
    food_left = bool(state[FOOD_LEFT])
    food_right = bool(state[FOOD_RIGHT])
    food_up = bool(state[FOOD_UP])
    food_down = bool(state[FOOD_DOWN])

    d = env.direction

    desired_dir = None

    if food_left and not food_right:
        desired_dir = Direction.LEFT
    elif food_right and not food_left:
        desired_dir = Direction.RIGHT
    elif food_up and not food_down:
        desired_dir = Direction.UP
    elif food_down and not food_up:
        desired_dir = Direction.DOWN
    else:
        return 0

    clock = [
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
        Direction.UP
    ]
    idx_curr = clock.index(d)
    idx_desired = clock.index(desired_dir)
    diff = (idx_desired - idx_curr) % 4

    if diff == 0:
        return 0  # already facing desired direction
    if diff == 1:
        return 1  # turn right
    if diff == 3:
        return 2  # turn left
    return 1


def untrained_q_agent(state: Tuple[int, ...], env: Any) -> int:
    """
    Baseline using an untrained Q-table (all zeros).
    This will always pick action 0 ('straight'), but we
    write it like this to match the Q-learning interface.
    """
    idx = state_tuple_to_int(state)
    return int(np.argmax(Q_ZERO[idx]))

AGENT_BASELINES: dict[str, Callable[[Tuple[int, ...], Any], int]] = {
    "random": random_agent,
    "safe_survival": safe_survival_agent,
    "greedy_food": greedy_food_agent,
    "untrained_q": untrained_q_agent,
}