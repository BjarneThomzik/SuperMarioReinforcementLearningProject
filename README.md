## Project Structure

- `src/` – contains all source code, organized into:
  - `reinforcement/` – approaches based on classical reinforcement learning (e.g., PPO).
  - `non_reinforcement/` – alternative methods (e.g., MDP, POMDP, Neuroevolution).

- `notebooks/` – Jupyter notebooks for training, experimentation, and evaluation.
- `runs/` – stores training artifacts such as models, videos, and plots from each training session.


## Dependencies

This project relies on the following open-source libraries:

- gym
- gym-super-mario-bros (https://github.com/Kautenja/gym-super-mario-bros)
- nes-py
- torch
- numpy
- matplotlib

All rights for the game "Super Mario Bros" belong to Nintendo. The gameplay videos in this repository are for academic use only and were created using the `gym-super-mario-bros` environment.
