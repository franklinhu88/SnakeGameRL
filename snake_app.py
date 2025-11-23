import pygame
import sys
import random
from snake_env import SnakeEnv, Direction  # your RL environment

class SnakeGameApp:
    def __init__(self):
        pygame.init()
        self.W, self.H = 640, 480
        self.display = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Snake RL / Human Mode")
        self.font = pygame.font.SysFont("arial", 40)
        self.small_font = pygame.font.SysFont("arial", 22)

    def draw_text(self, text, y, size=40):
        font = pygame.font.SysFont("arial", size)
        surface = font.render(text, True, (255,255,255))
        rect = surface.get_rect(center=(self.W//2, y))
        self.display.blit(surface, rect)

    # -------------------------
    # HOME SCREEN MENU
    # -------------------------
    def show_menu(self):
        while True:
            self.display.fill((0,0,0))
            self.draw_text("SNAKE RL", 120, size=60)
            self.draw_text("Press H to play as Human", 250)
            self.draw_text("Press A to watch the Agent", 300)
            self.draw_text("Press Q to Quit", 350)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        return "human"
                    if event.key == pygame.K_a:
                        return "agent"
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    # -------------------------
    # HUMAN PLAY MODE
    # -------------------------
    def run_human_mode(self):
        env = SnakeEnv(render_mode=True)
        env.reset()

        while True:
            # Default: keep going straight
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    # Escape to return to menu
                    if event.key == pygame.K_ESCAPE:
                        return

                    # Current direction (absolute)
                    d = env.direction

                    # --- ABSOLUTE CONTROLS WITH NO 180° TURN ---
                    if event.key == pygame.K_UP:
                        if d != Direction.DOWN:   # prevent reversal
                            # compute relative action that sets direction to UP
                            if d == Direction.LEFT:  action = 1
                            elif d == Direction.RIGHT: action = 2
                            else: action = 0

                    elif event.key == pygame.K_DOWN:
                        if d != Direction.UP:
                            if d == Direction.LEFT:  action = 2
                            elif d == Direction.RIGHT: action = 1
                            else: action = 0

                    elif event.key == pygame.K_LEFT:
                        if d != Direction.RIGHT:
                            if d == Direction.UP: action = 2
                            elif d == Direction.DOWN: action = 1
                            else: action = 0

                    elif event.key == pygame.K_RIGHT:
                        if d != Direction.LEFT:
                            if d == Direction.UP: action = 1
                            elif d == Direction.DOWN: action = 2
                            else: action = 0

            # Step with chosen action
            _, _, done, _ = env.step(action)
            env.render()

            if done:
                pygame.time.wait(800)
                env.reset()

    # -------------------------
    # AGENT PLAY MODE
    # -------------------------
    def run_agent_mode(self, agent=None):
        """
        agent: function(state) -> action (0,1,2)
        If agent=None, uses random policy.
        """
        env = SnakeEnv(render_mode=True)
        state = env.reset()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return  # back to home screen

            if agent:
                action = agent(state)
            else:
                action = random.choice([0,1,2])

            state, _, done, _ = env.step(action)
            env.render()

            if done:
                pygame.time.wait(600)
                state = env.reset()


# --------------------------------------
# MAIN APP LOOP
# --------------------------------------
if __name__ == "__main__":
    app = SnakeGameApp()
    while True:
        choice = app.show_menu()

        if choice == "human":
            app.run_human_mode()

        if choice == "agent":
            app.run_agent_mode(agent=None)  # random agent for now
