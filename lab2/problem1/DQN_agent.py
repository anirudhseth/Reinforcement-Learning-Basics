import numpy as np
import random
from collections import namedtuple, deque
from model import Neural_Network
import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    
    def __init__(self, state_dim, n_actions,seed,BUFFER_SIZE,BATCH_SIZE,DISCOUNT,TAU,LR,UPDATE_EVERY):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.seed = random.seed(seed)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.BUFFER_SIZE = BUFFER_SIZE 
        self.BATCH_SIZE = BATCH_SIZE         
        self.DISCOUNT = DISCOUNT            
        self.TAU = TAU            
        self.LR = LR              
        self.UPDATE_EVERY = UPDATE_EVERY       
        self.QNet = Neural_Network(self.state_dim, self.n_actions).to(self.device)
        self.QNet_target = Neural_Network(self.state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(self.n_actions, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.DISCOUNT)

    def act(self, state, eps=0.):
        if random.random() > eps:
          state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

          self.QNet.eval()
          with torch.no_grad():
            action_values = self.QNet(state)
          self.QNet.train()
          return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_target_max = self.QNet_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma*(Q_target_max)*(1-dones) 
        Q_expected = self.QNet(states).gather(1, actions) 
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.t_step == 0:
          self.update_target_network(self.QNet, self.QNet_target, self.TAU)              

    def update_target_network(self, source, target, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*source_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:

    def __init__(self, n_actions, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action