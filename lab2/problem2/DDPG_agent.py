
import numpy as np
from model import Actor,Critic
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import random

class Agent():
    def __init__(self, state_dim, action_dim, max_action, env,BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.max_action = max_action
        self.env = env
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA=GAMMA
        self.TAU=TAU
        self.NOISE=NOISE
        self.NOISE_CLIP=NOISE_CLIP
        self.POLICY_FREQUENCY=POLICY_FREQUENCY

    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))
        return action.clip(self.env.action_space.low, self.env.action_space.high)

    
    def train(self, replay_buffer,it):
      
       

          # Sample replay buffer 
          x, y, u, r, d = replay_buffer.sample(self.BATCH_SIZE)
          state = torch.FloatTensor(x).to(self.device)
          action = torch.FloatTensor(u).to(self.device)
          next_state = torch.FloatTensor(y).to(self.device)
          done = torch.FloatTensor(1 - d).to(self.device)
          reward = torch.FloatTensor(r).to(self.device)
          # state,action,reward,next_state,done = replay_buffer.sample()
          # Select action according to policy and add clipped noise 
          noise = torch.FloatTensor(u).data.normal_(0, self.NOISE).to(self.device)
          # noise = action.data.normal_(0, policy_noise).to(self.device)
          noise = noise.clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
          next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

          # Compute the target Q value
          target_Q= self.critic_target(next_state, next_action)
    
          target_Q = reward + (done * self.GAMMA * target_Q).detach()

          # Get current Q estimates
          current_Q = self.critic(state, action)

          # Compute critic loss
          critic_loss = F.mse_loss(current_Q, target_Q) 

          # Optimize the critic
          self.critic_optimizer.zero_grad()
          critic_loss.backward()
          self.critic_optimizer.step()

          # Delayed policy updates
          if it % self.POLICY_FREQUENCY == 0:

              # Compute actor loss
              actor_loss = -self.critic(state, self.actor(state)).mean()

              # Optimize the actor 
              self.actor_optimizer.zero_grad()
              actor_loss.backward()
              self.actor_optimizer.step()

              # Update the frozen target models
              for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                  target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
              for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                  target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        self.n_actions=n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples experiences from buffer
"""
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)
