import random
from collections import deque
from datetime import datetime

import cv2
import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import nn
import gym
from tqdm import tqdm

# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))

else:
    print("Device set to : cpu")
    device = torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.action_dim = action_dim

        conv1_size_h = (60 - 8) // 4 + 1
        conv1_size_w = (64 - 8) // 4 + 1
        conv2_size_h = (conv1_size_h - 4) // 2 + 1
        conv2_size_w = (conv1_size_w - 4) // 2 + 1
        flattened_size = conv2_size_h * conv2_size_w * 32

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=256),
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
            state = state.to(device)
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
        states = torch.cat(states).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).float().to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
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


# renders given frames with mediapy and shows a video
def renderEnv(frames):
    import mediapy as media
    media.show_video(frames, fps=60 // 4)


# plot for visualizing results
def plotRewardandTime(avg_norm_reward, avg_length):
    import matplotlib.pyplot as plt
    x = np.linspace(0, len(avg_reward), len(avg_reward))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    axs[0].plot(x, avg_norm_reward)
    axs[0].set_title("avg_norm_reward")

    axs[1].plot(x, avg_length)
    axs[1].set_title("avg_length")
    plt.show()


import gym


# This environment wrapper is used to stop a run if mario is stuck on a pipe
class DeadlockEnv(gym.Wrapper):
    def __init__(self, env, threshold=10):
        super().__init__(env)
        self.last_x_pos = 0
        self.count = 0
        self.threshold = threshold
        self.lifes = 3
        self.stage = 1
        self.world = 1

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info['x_pos']

        if x_pos <= self.last_x_pos:
            self.count += 1
        else:
            self.count = 0
            self.last_x_pos = x_pos

        if info['life'] != self.lifes or info["stage"] != self.stage or info["world"] != self.world:
            self.last_x_pos = x_pos
            self.count = 0
            self.lifes = info['life']
            self.stage = info["stage"]
            self.world = info["world"]

        if self.count >= self.threshold:
            reward = -15
            done = True

        return state, reward, done, info


# skipframe wrapper
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        reward_out = 0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            reward_out += reward
            if done:
                break
        reward_out /= max(1, i + 1)

        return obs, reward_out, done, info


# downsample wrapper to reduce dimensionality
def Downsample(ratio, state):
    (oldh, oldw, oldc) = state.shape
    newshape = (oldh // ratio, oldw // ratio, oldc)
    frame = cv2.resize(state, (newshape[0], newshape[1]), interpolation=cv2.INTER_AREA)
    return frame


# small function to change rgb images to grayscale
def GrayScale(state):
    return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)


from IPython.core.display import clear_output

frameskip = 4  # the frameskip value of the environment
down_sample_rate = 4  # downsample rate. Calculated as: original_dimension/down_sample_rate
frame_stack = 4  # frame stacking value

env = gym_super_mario_bros.make(
    'SuperMarioBros-v1')  # the environment. v0 is with original background, v1 has the background removed
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # The Joypadspace sets the available actions. We use SIMPLE_MOVEMENT.
env = SkipFrame(env, skip=frameskip)  # Skipframewrapper to skip some frames
env = DeadlockEnv(env, threshold=(
                                             60 * 2) // frameskip)  # Deadlock environment wrapper to stop the game if mario is stuck at a pipe

action_dim = env.action_space.n  # action space dimension
# state_dim = env.state_space.n  # Currently we flatten the input and therefore set the state_dim manually

dqn_agent = Agent(action_dim=action_dim, learning_rate=0.00025, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                  epsilon_decay=0.0000009, replay_buffer_size=1000000, replay_batch_size=32)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# some helper variables
time_step = 0
max_training_epochs = 10000
max_ep_len = 10000
update_timestep = max_ep_len

# We mount the google drive to save and load PPO states.
# from google.colab import drive
# drive.mount('/content/gdrive')

# If an agent is saved you can uncomment the following line to load the weights.
# ppo_agent.load("/content/gdrive/My Drive/ppo.save")

avg_reward_temp = []
avg_length_temp = []
avg_norm_reward_temp = []
avg_reward = []
avg_length = []
avg_norm_reward = []
updates = 0

episode_list = []

tbar = tqdm(range(1, max_training_epochs))
for i in tbar:
    # first we reset the state
    state = env.reset()
    state, reward, done, info = env.step(torch.randint(low=0, high=action_dim, size=(1,)).item())
    old_y = info['y_pos']
    current_ep_reward = 0
    # as we stack some frames, we create a buffer with empty frames for the first inputs
    states_buffer = [np.zeros((3840,)) for _ in range(3)]
    frames = []
    for _ in range(frame_stack):
        # Downsampling the environment
        in_state = GrayScale(Downsample(down_sample_rate, state.copy())) / 255

        # creating the new stack for the current frame
        states_buffer.append(in_state)
    states_input = torch.tensor(states_buffer[-frame_stack:]).unsqueeze(0)
    # the collection loop
    for t in range(1, max_ep_len):
        # selecting an action
        action = dqn_agent.choose_action(states_input)
        # print(np.asarray(states_buffer).flatten().min(),np.asarray(states_buffer).flatten().max())

        # performing the action and receiving the information from the environments
        state, reward, done, info = env.step(action)

        # reward shaping
        new_y = info['y_pos']
        if new_y > old_y:
            reward += 1
        old_y = new_y

        # Downsampling the environment
        in_state = GrayScale(Downsample(down_sample_rate, state.copy())) / 255

        # creating the new stack for the current frame
        states_buffer.append(in_state)
        new_states_input = torch.tensor(states_buffer[-frame_stack:]).unsqueeze(0)

        dqn_agent.add_to_replay_buffer((states_input, action, reward, new_states_input, done))
        dqn_agent.learn()
        updates += 1
        states_input = new_states_input

        # Every 10 epochs we render the environments and therefore save the state
        if not done:
            frames.append(state.copy())

        time_step += 1
        current_ep_reward += reward

        # every update_steps (2048) we update the algorithm
        # if time_step % update_steps == 0:

        # if the run is done we break the loop
        if done:
            break
        if len(frames) > 0:
            episode_list.append((current_ep_reward, frames))

        # We collect information every run and write them to the console
        avg_reward_temp.append(current_ep_reward)
        avg_length_temp.append(t)
        tbar.set_description("timestep: " + str(time_step) + " updates: " + str(updates) + " reward: " + str(
            np.asarray(avg_norm_reward_temp[-50:]).mean()))
        avg_norm_reward_temp.append(current_ep_reward / max(1, t))

        # Every 10 epochs we render the current environment
        if i % 10 == 0:

            avg_reward.append(np.median(avg_reward_temp))
            avg_length.append(np.median(avg_length_temp))
            avg_norm_reward.append(np.median(avg_norm_reward_temp))

            if len(episode_list) > 0:
                clear_output(wait=True)
                episode_list.sort(key=lambda x: x[0])  # we sort by the received reward and pick the best run to visualize

                print("--------------------------------")
                print("Epoch", i, "done:")
                print("Update iterations:", updates)
                print("Statistics:")
                print("")
                print("Reward of best episode:", episode_list[-1][0])
                print("Length of best episode:", len(episode_list[-1][1]))
                print("")
                print("Average total reward:", np.asarray(avg_reward[-50:]).mean())
                print("Average normalized reward:", np.asarray(avg_norm_reward[-50:]).mean())
                print("Average length:", np.asarray(avg_length[-50:]).mean())
                print("--------------------------------")
                plotRewardandTime(avg_norm_reward, avg_length)

                renderEnv(episode_list[-1][1])
                episode_list = []

                # ppo_agent.save("/content/gdrive/My Drive/ppo.save")

env.close()
