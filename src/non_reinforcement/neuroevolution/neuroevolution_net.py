import random

import torch.nn as nn
import torch


class NeuroevolutionNet(nn.Module):
    """Currently dummy class. Always outputs 1 or 2"""

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, x):
        logits = torch.zeros((x.shape[0], self.num_actions))
        logits[:, random.choice([1, 2])] = 1
        return logits
