
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self,n_states, n_actions):
        self.n_actions = n_actions
        self.n_states = n_states
        self.last_action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_net =myNetwork(self.n_states,self.n_actions)
        self.target_network = myNetwork(self.n_states,self.n_actions)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        state_tensor = torch.tensor([state], requires_grad=False,dtype=torch.float32)
        
        

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


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
