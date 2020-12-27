import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import Agent
import time

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def main():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = Agent(state_dim= env.observation_space.shape[0], n_actions=env.action_space.n,seed=0, BUFFER_SIZE = 100000, BATCH_SIZE=64,DISCOUNT=0.99,TAU=1,LR=7e-4,UPDATE_EVERY=4)
    n_ep_running_average = 50
    n_episodes=10
    max_t=1000
    eps_end=0.01
    eps_decay=0.995
    eps = 1.0                    # initialize epsilon
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    EPISODES = trange(n_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        state = env.reset()
        total_episode_reward = 0.
        t=0
        for k in range(max_t):
            action = agent.act(state, eps)
            #env.render()
            next_state, reward, done, _ = env.step(action)
            total_episode_reward += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            t+=1
            if done:
                break 
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))




if __name__ == "__main__":
    main()