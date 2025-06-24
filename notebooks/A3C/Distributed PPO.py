import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.multiprocessing as mp
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch.optim as optim
import collections
import gym
import cv2
import time
import os


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


def Downsample(ratio, state):
    (oldh, oldw, oldc) = state.shape
    newshape = (oldh // ratio, oldw // ratio, oldc)
    frame = cv2.resize(state, (newshape[0], newshape[1]), interpolation=cv2.INTER_AREA)
    return frame


def GrayScale(state):
    return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = DeadlockEnv(env, threshold=60)
    return env


class Actor_Critic(nn.Module):
    def __init__(self, env=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


def ppo_worker_fn(worker_id, shared_state_dict, queue, env_fn, n_steps=128, max_episodes=500000):
    """
    Worker-Funktion für parallele Datensammlung
    """
    try:
        env = env_fn()
        local_model = Actor_Critic()

        # Lade aktuelle Gewichte
        local_model.load_state_dict(dict(shared_state_dict))
        local_model.eval()

        state = env.reset()
        done = False
        episode_count = 0

        print(f"[Worker {worker_id}] Starting data collection")

        while episode_count < max_episodes:
            states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []

            for _ in range(n_steps):
                if done:
                    state = env.reset()
                    episode_count += 1
                    if episode_count >= max_episodes:
                        break

                # Preprocessing
                processed_state = GrayScale(Downsample(4, state)).flatten()
                state_tensor = torch.from_numpy(processed_state).float()

                with torch.no_grad():
                    action, logprob, value = local_model.act(state_tensor)

                next_state, reward, done, info = env.step(action.item())

                states.append(state_tensor)
                actions.append(action)
                logprobs.append(logprob)
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                dones.append(torch.tensor([done], dtype=torch.float32))
                values.append(value.squeeze(0))

                state = next_state

            if len(states) == 0:
                break


            with torch.no_grad():
                processed_next_state = GrayScale(Downsample(4, state)).flatten()
                next_tensor = torch.from_numpy(processed_next_state).float()
                next_value = local_model.critic(next_tensor) if not done else torch.tensor([0.0])

            traj = {
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "logprobs": torch.stack(logprobs),
                "rewards": torch.stack(rewards),
                "dones": torch.stack(dones),
                "values": torch.stack(values),
                "next_value": next_value,
                "worker_id": worker_id
            }


            queue.put(traj)


            local_model.load_state_dict(dict(shared_state_dict))

        env.close()
        print(f"[Worker {worker_id}] Finished data collection")

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()


def compute_advantages(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]

    return torch.stack(advantages)


def evaluate_model(model, env_fn, total_frames=4000, render=False, save_video=False, video_path="mario_gameplay.mp4"):
    """
    Modell evaluieren über feste Anzahl Frames (startet automatisch neu bei done=True)
    """
    env = env_fn()
    state = env.reset()
    done = False

    total_reward = 0
    episode_count = 0
    episode_rewards = []
    current_episode_reward = 0


    frames = []
    if save_video:
        print(f"Recording video... Will be saved as {video_path}")

    print(f"Starting evaluation for {total_frames} frames...")

    for frame in range(total_frames):
        # Preprocessing
        processed = GrayScale(Downsample(4, state)).flatten()
        tensor = torch.from_numpy(processed).float()


        with torch.no_grad():
            action, _, _ = model.act(tensor)


        state, reward, done, info = env.step(action.item())
        current_episode_reward += reward
        total_reward += reward


        if save_video:
            # Umwandlung von RGB zu BGR für OpenCV
            frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)


        if render:
            try:
                env.render()
                time.sleep(0.01)
            except:
                pass


        if done:
            episode_rewards.append(current_episode_reward)
            episode_count += 1
            print(
                f"Episode {episode_count} finished - Reward: {current_episode_reward:.2f} - Frame: {frame + 1}/{total_frames}")


            state = env.reset()
            current_episode_reward = 0
            done = False


    if current_episode_reward > 0:
        episode_rewards.append(current_episode_reward)
        episode_count += 1

    env.close()


    if save_video and frames:
        save_frames_as_video(frames, video_path, fps=30)


    avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
    print(f"\n[Evaluation Summary]")
    print(f"Total Frames: {total_frames}")
    print(f"Total Episodes: {episode_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward per Episode: {avg_episode_reward:.2f}")
    print(f"Average Reward per Frame: {total_reward / total_frames:.4f}")

    return {
        'total_reward': total_reward,
        'episode_rewards': episode_rewards,
        'avg_episode_reward': avg_episode_reward,
        'total_episodes': episode_count,
        'frames_played': total_frames
    }


def save_frames_as_video(frames, output_path, fps=30):
    """
    Frames als Video speichern
    """
    if not frames:
        print("No frames to save!")
        return

    try:
        import cv2


        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec


        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Saving {len(frames)} frames to {output_path}...")


        for i, frame in enumerate(frames):
            video_writer.write(frame)


            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(frames)} frames")

        video_writer.release()
        print(f"Video saved successfully: {output_path}")

    except Exception as e:
        print(f"Error saving video: {e}")


        import os
        frame_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(frame_dir, exist_ok=True)

        print(f"Saving frames as images in {frame_dir}/")
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{frame_dir}/frame_{i:06d}.png", frame)

        print(f"Frames saved as images. You can create video with:")
        print(f"ffmpeg -r {fps} -i {frame_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_path}")


def main():

    policy_model = Actor_Critic()


    manager = mp.Manager()
    shared_state_dict = manager.dict()
    shared_state_dict.update(policy_model.state_dict())


    queue = mp.Queue(maxsize=20)


    optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-4)

    # Worker starten
    num_workers = 10
    workers = []

    print(f"Starting {num_workers} workers...")

    for worker_id in range(num_workers):
        p = mp.Process(
            target=ppo_worker_fn,
            args=(worker_id, shared_state_dict, queue, make_env, 128, 25)  # 25 Episoden pro Worker
        )
        p.start()
        workers.append(p)


    update_count = 0
    total_updates = 100000  #später erhöhen

    print("Starting training loop...")

    try:
        while update_count < total_updates:
            try:

                traj = queue.get(timeout=30)

                # Extrahiere Daten
                states = traj["states"]
                actions = traj["actions"]
                old_logprobs = traj["logprobs"]
                rewards = traj["rewards"]
                dones = traj["dones"]
                values = traj["values"]
                next_value = traj["next_value"]
                worker_id = traj["worker_id"]


                advantages = compute_advantages(rewards, values, next_value, dones)
                returns = advantages + values


                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO Update
                for epoch in range(4):
                    logprobs, state_values, entropy = policy_model.evaluate(states, actions)

                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = ((returns - state_values) ** 2).mean()
                    entropy_bonus = entropy.mean()

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)  # Gradient clipping
                    optimizer.step()


                shared_state_dict.update(policy_model.state_dict())

                update_count += 1

                if update_count % 10 == 0:
                    avg_reward = rewards.mean().item()
                    print(
                        f"Update {update_count}/{total_updates} | Worker {worker_id} | Avg Reward: {avg_reward:.2f} | Loss: {loss:.4f}")


                    if update_count % 25 == 0:
                        eval_reward = evaluate_model(policy_model, make_env, render=False)
                        print(f"Evaluation Reward: {eval_reward:.2f}")

            except Exception as e:
                print(f"Error in training loop: {e}")  # Hier ist noch ein Bug drin
                break

    finally:

        print("Terminating workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()

        print("Training completed!")
        evaluate_model(policy_model, make_env, total_frames=4000,save_video=True, render=True)




if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()