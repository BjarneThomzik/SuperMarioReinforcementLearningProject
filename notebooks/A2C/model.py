import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def extract_features(self, state):
        return self.cnn(state)

    def act(self, state):  # state: (B, C, H, W)
        with torch.no_grad():
            features = self.extract_features(state)              # (B, feature_dim)
            probs = self.actor(features)                         # (B, action_dim)
            dist = Categorical(probs)
            actions = dist.sample()                              # (B,)
            log_probs = dist.log_prob(actions)                   # (B,)
            entropy = dist.entropy()
            return actions, log_probs, entropy

    def evaluate(self, states, actions):
        features = self.extract_features(states)
        probs = self.actor(features)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(features).squeeze(-1)
        return log_probs, entropy, values