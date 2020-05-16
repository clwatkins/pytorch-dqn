import datetime as dt
from pathlib import Path
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym

from agent import DQNAgent
from agent_args import AGENT_ARGS

import warnings
warnings.filterwarnings("ignore")


def train(
        training_episodes: int,
        env_name: str,
        agent_kwargs: dict,
        log_every: int = 500,
        render: bool = True
):

    env = gym.make(env_name)

    agent_kwargs['observation_space_dim'] = env.observation_space.shape[0]
    agent_kwargs['n_actions'] = env.action_space.n

    agent = DQNAgent(**agent_kwargs)

    print('Agents initialised. Training...')

    episode_rewards = []
    start_time = time.time()

    for episode in range(1, training_episodes + 1):

        obs = env.reset()
        done = False
        agent_losses = 0
        ep_reward = 0
        ep_step = 0

        while not done:
            action = agent.act(np.array(obs))

            next_obs, reward, done, info = env.step(action)
            env.render() if render and not episode % log_every else None

            ep_reward += reward

            loss = agent.step(
                state=np.array(obs),
                action=action,
                reward=reward,
                next_state=np.array(next_obs),
                done=done
            )

            agent_losses += loss if loss else 0

            obs = next_obs
            ep_step += 1

        episode_rewards.append(ep_reward)

        TB_WRITER.add_scalar('Loss', agent_losses, episode)
        TB_WRITER.add_scalar('Episode reward', ep_reward, episode)
        TB_WRITER.add_scalar('Epsilon', agent.epsilon, episode)

        if not episode % log_every:
            current_time = time.time()

            if render:
                time.sleep(0.2)  # pause to see final state

            print(f'Ep: {episode} / '
                  f'(Last {log_every:,.0f}) Mean: {np.mean(episode_rewards[-log_every:]):.1f} / '
                  f'Min: {np.min(episode_rewards[-log_every:]):.1f} / '
                  f'Max: {np.max(episode_rewards[-log_every:]):.1f} / '
                  f'EPS: {episode / (current_time - start_time):.1f} / '
                  f'Agent epsilon: {agent.epsilon:.2f}'
                  )

    print('Done training!\n')
    env.close()

    return agent, episode_rewards


def test(agent: DQNAgent, test_eps):
    env = gym.make(ENV_NAME)
    ep_rewards = []

    for test_ep in range(test_eps):
        obs = env.reset()
        done = False

        ep_reward = 0
        ep_step = 0

        while not done:

            action = agent.act(np.array(obs), evaluate=True)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            obs = next_obs

            ep_reward += reward
            ep_step += 1

        ep_rewards.append(ep_reward)
        time.sleep(0.2)

    print('\n')
    print('=== Test performance ===')
    print(f'Mean: {np.mean(ep_rewards):.1f} / '
          f'Min: {np.min(ep_rewards):.1f} / '
          f'Max: {np.max(ep_rewards):.1f}')

    env.close()
    return ep_rewards


if __name__ == '__main__':

    ENV_NAME = 'CartPole-v0'
    MODEL_NAME = 'DQN-LINx3-64'

    LOG_EVERY = 200
    PER_AGENT_REWARD = 1.0

    LOGGING_DEST = Path.cwd().joinpath(
        Path(f"logs/{MODEL_NAME}-{ENV_NAME}-{dt.datetime.now().strftime('%y%m%d-%H%M%S')}"))

    TB_WRITER = SummaryWriter(str(LOGGING_DEST))

    TRAINING_EPISODES = 1_000

    print('Beginning training')
    print('Logging to:', LOGGING_DEST)

    trained_agents, training_rewards = train(TRAINING_EPISODES, ENV_NAME, AGENT_ARGS,
                                             log_every=LOG_EVERY, render=True)
    test(trained_agents, 5)
