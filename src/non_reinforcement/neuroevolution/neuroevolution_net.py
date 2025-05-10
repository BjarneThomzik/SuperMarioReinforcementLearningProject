import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class NeuroevolutionNet(nn.Module):
    """
    A convolutional neural network combined with a multilayer perceptron
    used for policy selection in neuroevolution settings (Super Mario Bros.).

    This network takes grayscale game frames as input and outputs raw action logits.
    It supports mutation for evolutionary algorithms.

    Args:
        input_channels (int): Number of input channels (e.g. 1 for grayscale).
        num_actions (int): Number of possible output actions (logits).
        cnn_config (List[Dict]): List of dicts for each Conv2D layer (keys: out_channels, kernel_size, stride).
        mlp_config (List[int]): List of hidden layer sizes for the MLP.
    """

    def __init__(self, input_channels: int, num_actions: int, cnn_config: List[Dict], mlp_config: List[int]):
        super().__init__()
        self.num_actions = num_actions
        self.cnn_config = cnn_config
        self.mlp_config = mlp_config

        # Construct CNN from cnn_config
        cnn_layers = []
        in_channels = input_channels
        for layer_cfg in self.cnn_config:
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_cfg["out_channels"],
                kernel_size=layer_cfg["kernel_size"],
                stride=layer_cfg["stride"]
            ))
            cnn_layers.append(nn.ReLU())
            in_channels = layer_cfg["out_channels"]
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        # Construct MLP from mlp_config
        mlp_layers = []
        for in_f, out_f in zip(self.mlp_config[:-1], self.mlp_config[1:]):
            mlp_layers.append(nn.Linear(in_f, out_f))
            mlp_layers.append(nn.ReLU())
        # Remove the last ReLU layer so that we have a logit output layer
        mlp_layers.pop()
        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes convolutional and mlp layers using Xavier initialization.
        """
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, 84, 84).

        Returns:
            torch.Tensor: Raw output logits of shape (batch, num_actions).
        """
        x = self.cnn(x)
        x = self.mlp(x)
        return x

    def mutate(self, mutation_rate: float = 0.05, mutation_strength: float = 0.1):
        """
        Applies in-place random mutations to the model parameters by adding
        Gaussian noise to a fraction of the weights.

        Args:
            mutation_rate (float): Fraction of weights to mutate (range 0.0–1.0).
            mutation_strength (float): Standard deviation of the Gaussian noise.
        """
        with torch.no_grad():
            # Iterate over all trainable parameters and mutate about [mutation_rate] of all
            for param in self.parameters():
                if param.requires_grad:
                    mask = torch.rand_like(param) < mutation_rate
                    noise = torch.normal(mean=0.0, std=mutation_strength, size=param.shape, device=param.device)
                    param.add_(mask * noise)

    def save_model(self, path: str, input_channels: int, action_set: List[str]):
        """
        Saves the model weights and architecture metadata.

        Args:
            path (str): File path to save the model.
            input_channels (int): Number of input channels (e.g. 1 for grayscale).
            action_set (List[str]): Action list used in the training environment.
        """
        torch.save({
            "state_dict": self.state_dict(),
            "cnn_config": self.cnn_config,
            "mlp_config": self.mlp_config,
            "input_channels": input_channels,
            "num_actions": self.num_actions,
            "action_set": action_set,
        }, path)

    @staticmethod
    def load_model(path: str) -> Tuple["NeuroevolutionNet", List[str]]:
        """
        Loads a saved model and returns it along with its action set.

        Args:
            path (str): Path to the saved model file.

        Returns:
            Tuple[NeuroevolutionNet, List[str]]: Loaded model and action list.
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = NeuroevolutionNet(
            input_channels=checkpoint["input_channels"],
            num_actions=checkpoint["num_actions"],
            cnn_config=checkpoint["cnn_config"],
            mlp_config=checkpoint["mlp_config"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint.get("action_set", [])
