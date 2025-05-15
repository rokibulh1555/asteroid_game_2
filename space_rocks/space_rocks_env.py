import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from game import SpaceRocks

class SpaceRocksEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(SpaceRocksEnv, self).__init__()
        self.render_mode = render_mode
        self.game = SpaceRocks()
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(600, 800, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)  # [0: do nothing, 1: left, 2: right, 3: accelerate, 4: shoot]

    def reset(self, seed=None, options=None):
        self.game = SpaceRocks()
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        pygame.event.pump()

        # Apply action
        keys = [False] * 5
        if action == 1:
            self.game.spaceship.rotate(clockwise=False)
        elif action == 2:
            self.game.spaceship.rotate(clockwise=True)
        elif action == 3:
            self.game.spaceship.accelerate()
        elif action == 4:
            self.game.spaceship.shoot()

        self.game._process_game_logic()
        obs = self._get_observation()
        reward, terminated = self._calculate_reward()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        self.game._draw()

    def close(self):
        pygame.quit()

    def _get_observation(self):
        surface = self.game.screen
        return np.transpose(
            pygame.surfarray.array3d(surface), (1, 0, 2)
        )

    def _calculate_reward(self):
        if self.game.message == "You won!":
            return 100.0, True
        elif self.game.message == "You lost!":
            return -100.0, True
        else:
            return 0.1, False
