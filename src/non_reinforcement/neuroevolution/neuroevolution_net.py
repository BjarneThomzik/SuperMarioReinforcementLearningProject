import torch
import torch.nn as nn


class NeuroevolutionNet(nn.Module):
    """
    A convolutional neural network combined with a multilayer perceptron
    used for policy selection in neuroevolution settings (Super Mario Bros.).

    This network takes grayscale game frames as input and outputs raw action logits.
    It supports mutation for evolutionary algorithms.

    Args:
        input_channels (int): Number of input channels (usually 1 for grayscale).
        num_actions (int): Number of discrete actions

    Attributes:
        cnn (nn.Sequential): Convolutional feature extractor.
        mlp (nn.Sequential): Fully connected decision layers.
    """

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        # Create CNN and MLP
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

        # Initialize weights with Xavier initialization
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
