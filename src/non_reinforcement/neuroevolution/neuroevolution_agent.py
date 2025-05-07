import cv2
import numpy as np
import torch
from torch import Tensor
from gym import Env
from src.non_reinforcement.neuroevolution.neuroevolution_net import NeuroevolutionNet


class NeuroevolutionAgent:
    """Agent using a neuroevolutionary network to select actions and evaluate fitness."""

    def __init__(self, neuroevolution_net: NeuroevolutionNet, max_steps: int = 1000) -> None:
        """
        Initialize the agent.

        Args:
            neuroevolution_net (NeuroevolutionNet): The policy network used for action selection.
            max_steps (int): Maximum steps per evaluation episode.
        """
        self.neuroevolution_net: NeuroevolutionNet = neuroevolution_net
        self.fitness: float = 0.0
        self.max_steps: int = max_steps
        self._initial_lives: int = 1

    def act(self, processed_obs: Tensor) -> int:
        """
        Select an action based on the processed observation.

        Args:
            processed_obs (torch.Tensor): Input tensor of shape (1, 1, H, W).

        Returns:
            int: Index of the selected action.
        """
        with torch.no_grad():
            # Move input to same device as the model
            device = next(self.neuroevolution_net.parameters()).device
            processed_obs = processed_obs.to(device)

            # Get action from neural network
            logits = self.neuroevolution_net(processed_obs)
            action = torch.argmax(logits, dim=1).item()
        return action

    def evaluate(self, env: Env) -> float:
        """
        Evaluate the agent in a single episode and return the total reward.

        Args:
            env (gym.Env): Super Mario Gym environment.

        Returns:
            float: Total accumulated reward (fitness score).
        """
        # Get the first frame of the game and set the reward to 0
        observation = env.reset()
        total_reward = 0

        # Get initial lives
        _, _, _, info = env.step(0)
        self._initial_lives = info.get("life", 3)

        # Agent goes frame by frame through the game until he dies or the episode has ended
        for step in range(self.max_steps):
            processed_observation = self._preprocess(observation)
            action = self.act(processed_observation.unsqueeze(0))
            observation, reward, done, info = env.step(action)
            total_reward += reward

            # Break if the agent has died (once) or episode has ended
            if done or info.get("life", 3) < self._initial_lives:
                break

        self.fitness = total_reward
        return total_reward

    def _preprocess(self, obs: np.ndarray, size: int = 84) -> Tensor:
        """
        Convert an RGB image to a normalized grayscale tensor.

        Args:
            obs (np.ndarray): RGB input image.
            size (int): Target size for resizing (square).

        Returns:
            torch.Tensor: Preprocessed grayscale tensor with shape (1, H, W).
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (size, size))
        obs = obs / 255.0
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return tensor
