import os
import torch
import gym_super_mario_bros
from gym import Env
from gym.wrappers import RecordVideo
from nes_py.wrappers import JoypadSpace

from src.non_reinforcement.neuroevolution.neuroevolution_agent import NeuroevolutionAgent
from src.non_reinforcement.neuroevolution.neuroevolution_net import NeuroevolutionNet


class NeuroevolutionTrainer:
    """
    Trainer class to run a basic neuroevolution loop for training agents in the Super Mario Bros environment.
    Handles environment creation, agent evaluation, model mutation, and video recording.
    """

    def __init__(self, base_model: NeuroevolutionNet, env_name: str, action_set: list, device: torch.device,
                 video_dir: str, generations: int = 10, population_size: int = 10
                 ):
        """
        Initialize the neuroevolution trainer.

        Args:
            base_model (NeuroevolutionNet): The initial neural network model to evolve.
            env_name (str): Name of the Gym environment (e.g., "SuperMarioBros-v0").
            action_set (list): List of discrete actions to use (e.g., COMPLEX_MOVEMENT).
            device (torch.device): The device to run models on (e.g., CPU or CUDA).
            video_dir (str): Directory to save the evaluation video of the best agent.
            generations (int): Number of generations to evolve over.
            population_size (int): Number of agents per generation.
        """
        self.base_model = base_model.to(device)
        self.env_name = env_name
        self.action_set = action_set
        self.device = device
        self.video_dir = video_dir
        self.generations = generations
        self.population_size = population_size

        self.best_agent = None
        self.best_fitness = float('-inf')

        os.makedirs(self.video_dir, exist_ok=True)

    def make_env(self, record: bool = False) -> Env:
        """
        Create and return a new Mario Gym environment instance.

        Args:
            record (bool): Whether to record a video of the gameplay.

        Returns:
            Env: A wrapped Gym environment.
        """
        env = JoypadSpace(gym_super_mario_bros.make(self.env_name), self.action_set)
        if record:
            env = RecordVideo(env, self.video_dir, episode_trigger=lambda x: True)
        return env

    def clone_and_mutate(self, net: NeuroevolutionNet) -> NeuroevolutionNet:
        """
        Create a mutated clone of the given neural network.

        Args:
            net (NeuroevolutionNet): The source model to clone and mutate.

        Returns:
            NeuroevolutionNet: A new, mutated model on the same device.
        """
        new_net = NeuroevolutionNet(input_channels=1, num_actions=len(self.action_set)).to(self.device)
        new_net.load_state_dict(net.state_dict())
        new_net.mutate(mutation_rate=0.05, mutation_strength=0.1)
        return new_net

    def run(self):
        """
        Run the neuroevolution training loop across multiple generations.
        In each generation, agents are cloned, mutated, evaluated, and the best agent is tracked.
        """
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            agents = []
            for _ in range(self.population_size):
                model = self.clone_and_mutate(self.base_model)
                agent = NeuroevolutionAgent(model)
                agents.append(agent)

            for index, agent in enumerate(agents):
                env = self.make_env(record=False)
                fitness = agent.evaluate(env)
                env.close()
                print(f"Agent {index} Fitness: {fitness:.2f}")
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = agent
                    self.base_model = agent.neuroevolution_net
                    print(f"New best agent found! Fitness: {fitness:.2f}")

    def record_best_agent(self):
        """
        Play one episode using the best performing agent so far and save the gameplay video.
        The environment must support video recording.
        """
        if self.best_agent is None:
            print("No best agent available. Please run training first.")
            return

        env = self.make_env(record=True)
        print("\nRunning best agent for recording...")
        final_fitness = self.best_agent.evaluate(env)
        env.close()

        print(f"\nFinal fitness: {final_fitness:.2f}")
        print(f"Video saved to: {self.video_dir}")
