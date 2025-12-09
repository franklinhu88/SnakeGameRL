# snake_env.py
import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import sys

Point = namedtuple('Point', ['x', 'y'])

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLACK = (0, 0, 0)
GREEN1 = (0, 200, 50)
GREEN2 = (0, 255, 80)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeEnv:
    """
    Snake RL environment prepared for tabular Q-Learning.
    - Action space: {0: straight, 1: turn right, 2: turn left}
    - Observation: 11-element binary tuple:
        (danger_straight, danger_right, danger_left,
         dir_right, dir_left, dir_up, dir_down,
         food_left, food_right, food_up, food_down)
    - Rewards:
        +10  on eating food
        -10  on collision (terminal)
        -0.01 per step (time penalty)
    """
    def __init__(self, width=640, height=480, block_size=20, speed=10,
                 render_mode=False, seed=None):
        # grid geometry
        assert width % block_size == 0 and height % block_size == 0, "width/height must be divisible by block_size"
        self.w = width
        self.h = height
        self.block_size = block_size
        self.grid_w = width // block_size
        self.grid_h = height // block_size

        # Pygame rendering
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake RL Env")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 20)
            self.speed = speed

        # environment state
        self.direction = Direction.RIGHT
        self.head = None
        self.snake = None
        self.score = 0
        self.food = None
        self._rng = random.Random(seed)

        # internal
        self.frame_iteration = 0  # can be used to terminate very long episodes

        # init
        self.reset()

    # ----------------------------
    # Public API
    # ----------------------------
    def reset(self, seed=None):
        """Reset environment and return initial observation (tuple)."""
        if seed is not None:
            self._rng.seed(seed)
        self.direction = Direction.RIGHT
        # center head aligned to grid (use integer grid coordinates)
        center_x = (self.grid_w // 2) * self.block_size
        center_y = (self.grid_h // 2) * self.block_size
        self.head = Point(center_x, center_y)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - 2*self.block_size, self.head.y)
        ]
        self.score = 0
        self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def step(self, action):
        """
        action: 0 = straight, 1 = turn right, 2 = turn left (relative)
        returns: (state, reward, done, info)
        """
        self.frame_iteration += 1

        # handle events only when rendering, otherwise ignore pygame events
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()

        # apply action -> update direction
        self._update_direction(action)

        # move
        self._move()
        self.snake.insert(0, self.head)

        reward = -0.01  # small time penalty to encourage quicker solutions
        done = False

        # check collisions
        if self._is_collision():
            reward = -10
            done = True
            return self._get_state(), reward, done, {'score': self.score}

        # check food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # move by removing tail
            self.snake.pop()

        # cap episode length (avoid infinite wandering)
        if self.frame_iteration > 1000: 
            done = True

        return self._get_state(), reward, done, {'score': self.score}

    def render(self):
        """Render current state using pygame (only if render_mode=True)."""
        if not self.render_mode:
            return
        self.display.fill(BLACK)

        # draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            inner = 4
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+inner, pt.y+inner,
                                                             self.block_size-2*inner, self.block_size-2*inner))
        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        # draw score
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def close(self):
        if self.render_mode:
            pygame.quit()

    # ----------------------------
    # Internal utilities
    # ----------------------------
    def _place_food(self):
        """Place food on a free grid cell."""
        while True:
            x = self._rng.randint(0, self.grid_w - 1) * self.block_size
            y = self._rng.randint(0, self.grid_h - 1) * self.block_size
            p = Point(x, y)
            if p not in self.snake:
                self.food = p
                return

    def _move(self):
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        self.head = Point(x, y)

    def _is_collision(self, pt=None):
        """
        Check collision for point `pt` (if provided) or self.head.
        Collision if wall hit or runs into itself.
        """
        if pt is None:
            pt = self.head
        # boundary
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        # self collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_direction(self, action):
        """Convert relative action (straight/right/left) to absolute direction."""
        # order: RIGHT -> DOWN -> LEFT -> UP (clockwise)
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock.index(self.direction)
        if action == 0:  # straight
            new_dir = clock[idx]
        elif action == 1:  # turn right (clockwise)
            new_dir = clock[(idx + 1) % 4]
        elif action == 2:  # turn left (counter-clockwise)
            new_dir = clock[(idx - 1) % 4]
        else:
            raise ValueError("Invalid action. Must be 0,1,2")
        self.direction = new_dir

    # ----------------------------
    # State representation for tabular RL
    # ----------------------------
    def _get_state(self):
        """
        Compute compact state representation (tuple of 11 binary features):
        - danger_straight, danger_right, danger_left
        - direction one-hot: (right, left, up, down)
        - food location relative: (food_left, food_right, food_up, food_down)
        """
        head = self.head
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        # danger flags relative to current direction (straight, right, left)
        danger_straight = False
        danger_right = False
        danger_left = False

        if self.direction == Direction.RIGHT:
            danger_straight = self._is_collision(point_r)
            danger_right = self._is_collision(point_d)
            danger_left = self._is_collision(point_u)
        elif self.direction == Direction.LEFT:
            danger_straight = self._is_collision(point_l)
            danger_right = self._is_collision(point_u)
            danger_left = self._is_collision(point_d)
        elif self.direction == Direction.UP:
            danger_straight = self._is_collision(point_u)
            danger_right = self._is_collision(point_r)
            danger_left = self._is_collision(point_l)
        elif self.direction == Direction.DOWN:
            danger_straight = self._is_collision(point_d)
            danger_right = self._is_collision(point_l)
            danger_left = self._is_collision(point_r)

        # direction one-hot
        dir_right = int(self.direction == Direction.RIGHT)
        dir_left = int(self.direction == Direction.LEFT)
        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)

        # food location relative to head
        food_left = int(self.food.x < head.x)
        food_right = int(self.food.x > head.x)
        food_up = int(self.food.y < head.y)
        food_down = int(self.food.y > head.y)

        state = (
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            dir_right,
            dir_left,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down
        )
        return state

# ----------------------------
# Utilities for tabular Q-learning
# ----------------------------
def state_tuple_to_int(state_tuple):
    """
    Convert 11-bit tuple into an integer index for a Q-table.
    Order matches the _get_state tuple order.
    """
    idx = 0
    for bit in state_tuple:
        idx = (idx << 1) | (1 if bit else 0)
    return idx

