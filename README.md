# (D)DQN implementation in PyTorch

## Overview

A very simple implementation of the original DQN algorithm from [Minh et al 2015](https://www.nature.com/articles/nature14236), as well as the Double DQN (DDQN) subsequently developed by [van Hassalt et al 2015](https://arxiv.org/abs/1509.06461). 

Includes easily introspectable training scripts for simple Gym environments, as well as the Switch task found in [ma-gym](https://github.com/koulanurag/ma-gym).

- `agent.py`: network and agent definitions
- `agent_args.py`: simple dict for agent hyperparameters
- `train_cartpole.py`: single-agent Gym training loop
- `train_switch_simple.py`: multi-agent game from ma-gym. Train all agents concurrently on their own observations (highly unlikely to be successful)
- `train_switch_curriculum.py`: encourage Switch training by starting training for only one agent at a time, leaving untrained agents to a no-op
- `train_switch_joint_obs.py`: train agents concurrently, but on a joint observation space that concatenates all agent observations + a unique agent id

### Example of Switch4 trained by `train_switch_joint_obs.py`
![](https://s6.gifyu.com/images/switch4.gif)

## Usage

Just a simple `$ python train_cartpole.py`.

Edit hyperparams in `agent_args.py`.

## Requirements

- Python 3.6+
- PyTorch 1.4+
- Gym