import os
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
                 video_dir: str, generations: int = 10, population_size: int = 10, max_steps_per_episode: int = 1000,
                 mutation_rate: float = 0.05, mutation_strength: float = 0.1
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
        self.max_steps_per_episode = max_steps_per_episode
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

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

    def run(self):
        """
        Run the neuroevolution training loop across multiple generations.
        In each generation, agents are cloned, mutated, evaluated, and the best agent is tracked.
        After each generation, training metrics are updated and plotted live.
        """
        for generation in range(self.generations):
            agents = []
            for _ in range(self.population_size):
                model = self.clone_and_mutate(self.base_model)
                agent = NeuroevolutionAgent(model, self.max_steps_per_episode)
                agents.append(agent)

            fitnesses = []
            for index, agent in enumerate(agents):
                env = self.make_env(record=False)
                fitness = agent.evaluate(env)
                env.close()
                fitnesses.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = agent
                    self.base_model = agent.neuroevolution_net

            # Save metrics of the current generation
            gen_best = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            self.metrics_log.append({
                "generation": generation + 1,
                "best": gen_best,
                "avg": avg_fitness,
                "min": min_fitness
            })

            # Plot metrics of the current generation
            self.plot_metrics()

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
