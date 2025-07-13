import random
from collections import deque

import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_dim),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, action_dim, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, replay_buffer_size,
                 replay_batch_size):
        self.dqn = DQN(action_dim).to(device)
        self.optimizer = torch.optim.RMSprop(self.dqn.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_batch_size = replay_batch_size

    def choose_action(self, state):
        if torch.rand(1) <= self.epsilon:
            action = torch.randint(low=0, high=self.dqn.action_dim, size=(1,))
        else:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.dqn(state)
            action = torch.argmax(q_values)
        return action.item()

    def anneal_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def learn(self):
        if len(self.replay_buffer) < self.replay_batch_size:
            return
        batch = random.sample(self.replay_buffer, self.replay_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        q_values = self.dqn(states)
        with torch.no_grad():
            next_q_values = self.dqn(next_states)
            max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        target = q_values.detach().clone()
        target[range(len(actions)), actions] = target_q_values
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()