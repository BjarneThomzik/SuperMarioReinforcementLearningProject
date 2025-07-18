import copy
import math
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
                 directory: str, generations: int, population_size: int, max_steps_per_episode: int,
                 mutation_rate_range: tuple[float, float], mutation_strength_range: tuple[float, float],
                 roulette_wheel_selection_temperature: float, elitism: int, mutation_rate_sigmoid_decay: dict = None,
                 mutation_strength_sigmoid_decay: dict = None):
        """
        Initialize the neuroevolution trainer.

        Args:
            base_model (NeuroevolutionNet): The initial neural network model to evolve.
            env_name (str): Name of the Gym environment (e.g., "SuperMarioBros-v0").
            action_set (list): List of discrete actions to use (e.g., COMPLEX_MOVEMENT).
            device (torch.device): The device to run models on (e.g., CPU or CUDA).
            directory (str): Directory to save the evaluation video of the best agent.
            generations (int): Number of generations to evolve over.
            population_size (int): Number of agents per generation.
        """
        self.base_model = base_model.to(device)
        self.env_name = env_name
        self.action_set = action_set
        self.device = device
        self.directory = directory
        self.generations = generations
        self.population_size = population_size
        self.max_steps_per_episode = max_steps_per_episode
        self.mutation_rate_range = mutation_rate_range
        self.mutation_strength_range = mutation_strength_range
        self.roulette_wheel_selection_temperature = roulette_wheel_selection_temperature
        self.elitism = elitism
        self.mutation_rate_sigmoid_decay = mutation_rate_sigmoid_decay
        self.mutation_strength_sigmoid_decay = mutation_strength_sigmoid_decay

        self.best_agent = None
        self.best_fitness = float('-inf')
        self.metrics_log = []
        self.output_dir = None

        os.makedirs(self.directory, exist_ok=True)

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
            env = RecordVideo(env, self.directory, episode_trigger=lambda x: True)
        return env

    def clone_and_mutate(self, net: NeuroevolutionNet, generation: int) -> NeuroevolutionNet:
        """
        Clone a model and apply random mutation within specified ranges.

        Args:
            net (NeuroevolutionNet): The source model to clone and mutate.

        Returns:
            NeuroevolutionNet: A new, mutated model on the same device.
        """
        new_net = NeuroevolutionNet(input_channels=1, num_actions=len(self.action_set),
                                    cnn_config=self.base_model.cnn_config, mlp_config=self.base_model.mlp_config).to(
            self.device)
        new_net.load_state_dict(net.state_dict())

        if self.mutation_rate_sigmoid_decay:
            a, b, c = self.mutation_rate_sigmoid_decay["a"], self.mutation_rate_sigmoid_decay["b"], \
                self.mutation_rate_sigmoid_decay["c"]
            rate = self._sigmoid_decay(generation, a, b, c)
        else:
            rate = np.random.uniform(*self.mutation_rate_range)

        if self.mutation_strength_sigmoid_decay:
            a, b, c = self.mutation_strength_sigmoid_decay["a"], self.mutation_strength_sigmoid_decay["b"], \
                self.mutation_strength_sigmoid_decay["c"]
            strength = self._sigmoid_decay(generation, a, b, c)
        else:
            strength = np.random.uniform(*self.mutation_strength_range)

        new_net.mutate(rate, strength)
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
        logits = norm_fitnesses / self.roulette_wheel_selection_temperature
        probabilities = scipy.special.softmax(logits)

        # Sample parent index based on probabilities
        index = np.random.choice(len(agents), p=probabilities)
        return agents[index].neuroevolution_net

    def _create_population(self, agents, fitnesses, generation: int):
        """
        Creates a new population via fitness-proportional selection and elitism.
        Elitism: Keep the self.elitism number of the best agents without mutation.
        """
        new_agents = []

        # Copy the best agents of this generation (the elites)
        if self.elitism:
            elite_idx = np.argsort(fitnesses)[-self.elitism:]
            for idx in elite_idx:
                elite_net = copy.deepcopy(agents[idx].neuroevolution_net).to(self.device)
                new_agents.append(
                    NeuroevolutionAgent(elite_net, self.max_steps_per_episode)
                )

        # Select the rest of the agents with fitness proportionate selection and mutate them
        while len(new_agents) < self.population_size:
            parent_net = self._select_parent(agents, fitnesses)
            child_net = self.clone_and_mutate(parent_net, generation)
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
            agents = self._create_population(agents, fitnesses, generation)

    def record_best_agent(self):
        """
        Play one episode using the best performing agent so far and save the gameplay video.
        The environment must support video recording.
        """
        if self.best_agent is None:
            print("No best agent available. Please run training first.")
            return

        # Create base folder name from hyperparameters
        base_name = (
            f"max_fit_{int(self.best_fitness)}__"
            f"avg_fit_{int(np.mean([m['avg'] for m in self.metrics_log]))}__"
            f"env_name_{self.env_name}__"
            f"action_set_{len(self.action_set)}"
        )

        # Create unique output directory for this run
        run_dir = os.path.join(self.directory, base_name)
        i = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(self.directory, f"{base_name} ({i})")
            i += 1
        os.makedirs(run_dir)

        # Create recording environment that writes to this folder
        env = JoypadSpace(gym_super_mario_bros.make(self.env_name), self.action_set)
        env = RecordVideo(env, video_folder=run_dir, name_prefix="best_agent", episode_trigger=lambda x: True)

        print("\nRunning best agent for recording...")
        final_fitness = self.best_agent.evaluate(env)
        env.close()

        # Rename video to cleaner name
        original_path = os.path.join(run_dir, "best_agent-episode-0.mp4")
        final_path = os.path.join(run_dir, f"fitness_{self.best_fitness}.mp4")
        if os.path.exists(original_path):
            os.rename(original_path, final_path)

        # Delete the .meta.json file
        meta_path = os.path.join(run_dir, "best_agent-episode-0.meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)

        print(f"\nFinal fitness: {final_fitness:.2f}")
        print(f"Video saved to: {final_path}")

        self.output_dir = run_dir

    def plot_metrics(self):
        """
        Plot best, average, and minimum fitness across all completed generations.
        Displays configured hyperparameters in the plot title.
        Clears previous output to keep the notebook clean (for Jupyter).
        """
        if not self.metrics_log:
            print("No metrics to plot.")
            return

        generations = [m["generation"] for m in self.metrics_log]
        best = [m["best"] for m in self.metrics_log]
        avg = [m["avg"] for m in self.metrics_log]
        min_ = [m["min"] for m in self.metrics_log]

        # Compose title with hyperparameters
        decay_info = self._get_decay_description()
        title = (
            f"Env: {self.env_name} | Actions: {len(self.action_set)} | Gens: {self.generations} | Pop: {self.population_size} | "
            f"MaxSteps: {self.max_steps_per_episode} | "
            f"MutRateRange: {self.mutation_rate_range} | MutStrRange: {self.mutation_strength_range} | "
            f"RouletteWheelTemp: {self.roulette_wheel_selection_temperature} | Elitism: {self.elitism}"
            f"\n{decay_info}"
        )

        clear_output()

        plt.figure(figsize=(12, 6))
        plt.plot(generations, best, label="Best Fitness", color="green", linewidth=1)
        plt.plot(generations, avg, label="Average Fitness", color="blue", linewidth=3)
        plt.plot(generations, min_, label="Min Fitness", color="red", linewidth=1)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(title, fontsize=9)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_metrics(self):
        """
        Save best, average, and minimum fitness across all completed generations.
        Displays configured hyperparameters in the plot title.
        Clears previous output to keep the notebook clean (for Jupyter).
        """
        if not self.metrics_log:
            print("No metrics to save.")
            return

        generations = [m["generation"] for m in self.metrics_log]
        best = [m["best"] for m in self.metrics_log]
        avg = [m["avg"] for m in self.metrics_log]
        min_ = [m["min"] for m in self.metrics_log]

        # Compose title with hyperparameters
        decay_info = self._get_decay_description()
        title = (
            f"Env: {self.env_name} | Actions: {len(self.action_set)} | Gens: {self.generations} | Pop: {self.population_size} | "
            f"MaxSteps: {self.max_steps_per_episode} | "
            f"MutRateRange: {self.mutation_rate_range} | MutStrRange: {self.mutation_strength_range} | "
            f"RouletteWheelTemp: {self.roulette_wheel_selection_temperature} | Elitism: {self.elitism}"
            f"\n{decay_info}"
        )

        plt.figure(figsize=(12, 6))
        plt.plot(generations, best, label="Best Fitness", color="green", linewidth=1)
        plt.plot(generations, avg, label="Average Fitness", color="blue", linewidth=3)
        plt.plot(generations, min_, label="Min Fitness", color="red", linewidth=1)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(title, fontsize=9)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "fitness_plot.png"), dpi=300)
        plt.close()
        print(f"Fitness plot saved to {os.path.join(self.output_dir, 'fitness_plot.png')}")

    def save_best_model(self):
        """
        Saves the best model found during training.
        """

        if self.best_agent is None:
            print("No best agent available to save.")
            return

        model = self.best_agent.neuroevolution_net
        directory = f"{self.output_dir}/best_model_{self.best_fitness}.pt"
        model.save_model(directory, input_channels=1, action_set=self.action_set)
        print(f"Best model saved to {directory}")

    def load_model(self, path: str):
        """
        Loads a saved model and sets it as the new base_model.

        Args:
            path (str): Path to the saved model file.
        """
        model, action_set = NeuroevolutionNet.load_model(path)
        self.base_model = model
        self.action_set = action_set
        print(f"Loaded model from {path} as base model.")

    @staticmethod
    def _sigmoid_decay(x: int, a: float, b: float, c: float) -> float:
        """
        Compute the value of a shifted sigmoid decay function.

        Args:
            x (int): Input value (e.g., step, iteration, or epoch).
            a (float): Vertical shift applied to the curve.
            b (float): Controls the steepness of the curve (positive values produce a decay).
            c (float): Horizontal shift (the x-value at the inflection point).

        Returns:
            float: The value of the sigmoid decay function at x.
        """
        return 1 / (1 + math.exp(b * (x - c))) + a

    def _get_decay_description(self) -> str:
        """
        Returns a short string describing the configured sigmoid decay functions
        for mutation rate and strength, or "No decay" if none are set.
        """

        def format_sigmoid(name: str, params: dict) -> str:
            a, b, c = params["a"], params["b"], params["c"]
            return f"{name}: f(x)=1/(1+e^({b:.3f}*(x-{c:.0f}))) + {a:.3f}"

        parts = []

        if self.mutation_rate_sigmoid_decay:
            parts.append(format_sigmoid("RateDecay", self.mutation_rate_sigmoid_decay))
        if self.mutation_strength_sigmoid_decay:
            parts.append(format_sigmoid("StrengthDecay", self.mutation_strength_sigmoid_decay))

        return " | ".join(parts) if parts else "No decay"
