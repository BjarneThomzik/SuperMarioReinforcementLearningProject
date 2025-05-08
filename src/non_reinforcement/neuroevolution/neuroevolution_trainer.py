import copy
import os

import numpy as np
import scipy
import torch
import gym_super_mario_bros
from IPython.core.display_functions import clear_output
from gym import Env
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace

from src.non_reinforcement.neuroevolution.neuroevolution_agent import NeuroevolutionAgent
from src.non_reinforcement.neuroevolution.neuroevolution_net import NeuroevolutionNet


class NeuroevolutionTrainer:
    """
    Trainer class to run a basic neuroevolution loop for training agents in the Super Mario Bros environment.
    Handles environment creation, agent evaluation, model mutation, and video recording.
    """

    def __init__(self, base_model: NeuroevolutionNet, env_name: str, action_set: list, device: torch.device,
                 video_dir: str, generations: int, population_size: int, max_steps_per_episode: int,
                 mutation_rate: float, mutation_strength: float, wheel_selection_temperature: float):
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
        self.max_steps_per_episode = max_steps_per_episode
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.wheel_selection_temperature = wheel_selection_temperature

        self.best_agent = None
        self.best_fitness = float('-inf')
        self.metrics_log = []

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
        new_net = NeuroevolutionNet(input_channels=1, num_actions=len(self.action_set),
                                    cnn_config=self.base_model.cnn_config, mlp_config=self.base_model.mlp_config).to(
            self.device)
        new_net.load_state_dict(net.state_dict())
        new_net.mutate(self.mutation_rate, self.mutation_strength)
        return new_net

    def _select_parent(self, agents, fitnesses):
        """
        Selects a parent via softmax-weighted roulette selection.
        Fitness is normalized and passed through softmax with temperature to avoid domination of a single agent.
        """
        fitnesses = np.array(fitnesses, dtype=np.float32)

        # Normalize fitness values to [0, 1]
        min_f, max_f = fitnesses.min(), fitnesses.max()
        if max_f - min_f < 1e-6:
            norm_fitnesses = np.ones_like(fitnesses)
        else:
            norm_fitnesses = (fitnesses - min_f) / (max_f - min_f)

        # Apply softmax with temperature
        logits = norm_fitnesses / self.wheel_selection_temperature
        probabilities = scipy.special.softmax(logits)

        # Sample parent index based on probabilities
        index = np.random.choice(len(agents), p=probabilities)
        return agents[index].neuroevolution_net

    def _create_population(self, agents, fitnesses):
        """
        Creates a new population via fitness-proportional selection and mutation.
        """
        new_agents = []
        for _ in range(self.population_size):
            parent_net = self._select_parent(agents, fitnesses)
            child_net = copy.deepcopy(parent_net).to(self.device)
            child_net.mutate(self.mutation_rate, self.mutation_strength)
            new_agents.append(NeuroevolutionAgent(child_net, self.max_steps_per_episode))
        return new_agents

    def run(self):
        """
        Run the neuroevolution training loop across multiple generations.
        In each generation, agents are cloned, mutated, evaluated, and the best agent is tracked.
        After each generation, training metrics are updated and plotted live.
        """
        # The first generation uses copies from the base model
        agents = [
            NeuroevolutionAgent(copy.deepcopy(self.base_model), self.max_steps_per_episode)
            for _ in range(self.population_size)
        ]

        for generation in range(self.generations):
            fitnesses = []
            for agent in agents:
                env = self.make_env(record=False)
                fitness = agent.evaluate(env)
                env.close()
                fitnesses.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = agent

            # Save metrics of the current generation
            self.metrics_log.append({
                "generation": generation + 1,
                "best": max(fitnesses),
                "avg": float(np.mean(fitnesses)),
                "min": min(fitnesses)
            })
            self.plot_metrics()

            # Create new population with fitness proportional selection
            agents = self._create_population(agents, fitnesses)

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

    def plot_metrics(self):
        """
        Plot best, average, and minimum fitness across all completed generations.
        Clears previous output to keep the notebook clean (for Jupyter).
        """
        if not self.metrics_log:
            print("No metrics to plot.")
            return

        generations = [m["generation"] for m in self.metrics_log]
        best = [m["best"] for m in self.metrics_log]
        avg = [m["avg"] for m in self.metrics_log]
        min_ = [m["min"] for m in self.metrics_log]

        # Clear output
        clear_output()

        plt.figure(figsize=(10, 5))
        plt.plot(generations, best, label="Best Fitness", color="green", linewidth=1)
        plt.plot(generations, avg, label="Average Fitness", color="blue", linewidth=5)
        plt.plot(generations, min_, label="Min Fitness", color="red", linewidth=1)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Neuroevolution Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print global metrics
        last = self.metrics_log[-1]
        print(f"Generation: {last['generation']} / {self.generations}")
        print(f"\nAlltime best: {self.best_fitness:.2f}")
