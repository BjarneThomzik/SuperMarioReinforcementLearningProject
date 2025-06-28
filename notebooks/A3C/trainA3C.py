
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch.optim as optim
import collections
import torch


import gym
import cv2

#This environment wrapper is used to stop a run if mario is stuck on a pipe
class DeadlockEnv(gym.Wrapper):
    def __init__(self, env, threshold=20):
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

#skipframe wrapper
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
        reward_out /= max(1,i+1)

        return obs, reward_out, done, info

#downsample wrapper to reduce dimensionality
def Downsample(ratio,state):
  (oldh, oldw, oldc) = state.shape
  newshape = (oldh//ratio, oldw//ratio, oldc)
  frame = cv2.resize(state, (newshape[0], newshape[1]), interpolation=cv2.INTER_AREA)
  return frame

#small function to change rgb images to grayscale
def GrayScale(state):
  return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = DeadlockEnv(env, threshold=(60*2)//4)
    return env



class Actor_Critic:
    def __init__(self, env):
        self.env = env

        self.actor = nn.Sequential(
                            nn.Linear(3840, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 7),
                            nn.Softmax(dim=-1)
                        )
        self.critic = nn.Sequential(
                        nn.Linear(3840, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def act(self, state):
        # if we have a continuous action space we sample from a multivariate normal distribution
        # otherwise we calculate a categorical action space
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


import torch.multiprocessing as mp

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()


def worker_fn(worker_id, global_model, optimizer, env_fn, gamma=0.99, n_steps=10):
    print("test2")
    import time
    local_model = Actor_Critic(None)  # Lokale Kopie, ohne `env`
    local_model.actor.load_state_dict(global_model.actor.state_dict())
    local_model.critic.load_state_dict(global_model.critic.state_dict())

    env = env_fn()
    state = env.reset()
    done = False

    states, actions, rewards, logprobs = [], [], [], []

    #while True:
    for i in range(15000):
        if i % 100 == 0:
            print(i)
        if done:
            state = env.reset()

        processed_state = GrayScale(Downsample(4, state)).flatten()
        processed_tensor = torch.from_numpy(processed_state).float()

        action, logprob = local_model.act(processed_tensor)
        next_state, reward, done, _ = env.step(action.item())
        next_helper = next_state.copy()

        states.append(processed_tensor.unsqueeze(0))
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(torch.tensor([reward], dtype=torch.float32))
        state = next_helper

        if len(rewards) >= n_steps or done:
            with torch.no_grad():
                next_state_processed = GrayScale(Downsample(4, state)).flatten()
                next_tensor = torch.from_numpy(next_state_processed).float()
                next_value = local_model.critic(next_tensor) if not done else torch.tensor([[0.0]])

            returns = []
            R = next_value
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            states_tensor = torch.cat(states)
            actions_tensor = torch.stack(actions)
            logprobs_tensor = torch.stack(logprobs)
            returns_tensor = torch.cat(returns).detach()

            new_logprobs, values, entropy = local_model.evaluate(states_tensor, actions_tensor)
            advantage = returns_tensor - values.squeeze()

            policy_loss = -(new_logprobs * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            entropy_bonus = entropy.mean()
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()

            #Gradienten auf das Modell
            for local_param, global_param in zip(local_model.actor.parameters(), global_model.actor.parameters()):
                global_param._grad = local_param.grad
            for local_param, global_param in zip(local_model.critic.parameters(), global_model.critic.parameters()):
                global_param._grad = local_param.grad

            optimizer.step()

            #global syncen
            local_model.actor.load_state_dict(global_model.actor.state_dict())
            local_model.critic.load_state_dict(global_model.critic.state_dict())

            # Reset buffers
            states, actions, rewards, logprobs = [], [], [], []

        if worker_id == 0 and time.time() % 10 < 0.1:
            print(f"[Worker {worker_id}] Running...")
        # Optional: env.render()

def evaluate_model(model, env_fn, num_frames=400, render=True):
    import time
    env = env_fn()
    state = env.reset()
    done = False
    frame_count = 0
    total_reward = 0

    while frame_count < num_frames:
        if done:
            state = env.reset()

        processed = GrayScale(Downsample(4, state)).flatten()
        tensor = torch.from_numpy(processed).float()

        with torch.no_grad():
            action, _ = model.act(tensor)

        state, reward, done, _ = env.step(action.item())
        total_reward += reward

        if render:
            env.render()
            time.sleep(0.02)  # für menschliche Sichtbarkeit

        frame_count += 1

    env.close()
    print(f"\n[Evaluation] Total Reward over {num_frames} frames: {total_reward:.2f}")




if __name__ == "__main__":
    mp.set_start_method('spawn')  # wichtig für PyTorch
    global_model = Actor_Critic(None)
    global_model.actor.share_memory()
    global_model.critic.share_memory()

    optimizer = SharedAdam(list(global_model.actor.parameters()) + list(global_model.critic.parameters()), lr=1e-4)

    processes = []
    print("test1")
    for i in range(10):  # Zwei parallele Worker
        p = mp.Process(target=worker_fn, args=(i, global_model, optimizer, make_env))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    evaluate_model(global_model, make_env, num_frames=1000)
