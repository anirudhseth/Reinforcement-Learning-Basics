import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random

class Bank:
    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

  
    BANK_REWARD = 1
    POLICE_REWARD = -10
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze):
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.possible_actions         = self.__possible_actions()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.memory = dict()

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        s = 0;
        for player_i in range(self.maze.shape[0]):
            for player_j in range(self.maze.shape[1]):
                    for minotaur_i in range(self.maze.shape[0]):
                        for minotaur_j in range(self.maze.shape[1]):
                            minotaur_pos=(minotaur_i,minotaur_j)
                            player_pos=(player_i,player_j)
                            temp = (player_pos,minotaur_pos);
                            states[s] = temp
                            map[temp] = s;
                            s += 1;
        return states, map

    def __possible_actions(self):
        possible_actions = dict()
        for s_index, s in self.states.items():
            l = []
            for a, a_delta in self.actions.items():
                player_row = s[0][0]
                player_col = s[0][1]
                row = player_row + a_delta[0];
                col = player_col + a_delta[1];
                hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1])
                if(not hitting_maze_walls):
                    l.append(a)
                possible_actions[s_index] = l

        return possible_actions

    def epsilonGreedy(self, s_index, eps, Q):
        rand = np.random.uniform()
        if rand < eps:
            return random.choice(self.possible_actions[s_index])
        else:
            return np.argmax(Q[s_index, :])

    def reward(self, state, move_possible):
        r = 0
        if self.states[state][0]==self.states[state][1]:
            r = self.POLICE_REWARD
        elif self.maze[self.states[state][0]] == 1:
            r = self.BANK_REWARD
        if not move_possible:
            r = self.IMPOSSIBLE_REWARD
        return r

    def possibleMoves(self, state, action):
        checkMemory = self.memory.get((state, action), None)
        if checkMemory is not None:
            return checkMemory

        player_row = self.states[state][0][0]
        player_col = self.states[state][0][1]
        row = player_row + self.actions[action][0]
        col = player_col + self.actions[action][1]
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1])
        if hitting_maze_walls:
            player_pos_temp= (player_row,player_col);
        else:
            player_pos_temp=(row,col)
        police_actions = {4: (0, 0), 0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        next_states = []
        police_row = self.states[state][1][0]
        police_col = self.states[state][1][1]
        for police_action in police_actions:
            row_p = police_row + police_actions[police_action][0]
            col_p = police_col + police_actions[police_action][1]
            hitting_maze_walls_m =  (row_p == -1) or (row_p == self.maze.shape[0]) or \
                        (col_p == -1) or (col_p == self.maze.shape[1]) 
            if not hitting_maze_walls_m:
                police_pos_temp= (row_p,col_p);
                temp = (player_pos_temp,police_pos_temp);
                next_states.append((self.map[temp],self.reward(self.map[temp],not(hitting_maze_walls_m))))
        self.memory[(state, action)] = next_states
        return next_states

    def move(self, state, action):
        next_s, reward = random.choice(self.possibleMoves(state, action))
        return next_s, reward

    def draw_maze(self, maze,actions):
        # Some colours
        LIGHT_RED    = '#FFC4CC';
        LIGHT_GREEN  = '#95FD99';
        BLACK        = '#000000';
        WHITE        = '#FFFFFF';
        LIGHT_PURPLE = '#E8D0FF';
        LIGHT_ORANGE = '#FAE0C3';
        # Map a color to each cell in the maze
        col_map = {0: WHITE, 1: LIGHT_GREEN, 2: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED}
        # Give a color to each cell
        rows, cols = maze.shape
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

        # Create figure of the size of the city
        fig = plt.figure(1, figsize=(cols, rows))

        # Remove the axis ticks and add title title
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        rows, cols = maze.shape
        colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols,rows))

        # Create a table to color
        grid = plt.table(cellText=None,
                                cellColours=colored_maze,
                                cellLoc='center',
                                loc=(0, 0),
                                edges='closed')

        def plotActions(pos, action):
            cell = grid[pos]
            size = cell.get_width()*0.25
            mid_x = cell.get_x() + 0.5*cell.get_width()
            mid_y = cell.get_y() + 0.5*cell.get_height()
            dirs = dict()
            dirs[self.STAY] = (0,0)
            dirs[self.MOVE_DOWN] = (0, -size)
            dirs[self.MOVE_UP] = (0, size)
            dirs[self.MOVE_RIGHT] = (size, 0)
            dirs[self.MOVE_LEFT] = (-size, 0)
            x, y = dirs[action]
            return plt.arrow(mid_x - x/2, mid_y - y/2, x, y, width = 0.008)
     
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows)
            cell.set_width(1.0/cols)

        grid._do_cell_alignment()
        for pos, action in actions.items():
                plotActions(pos, action)

    def optimalPolicy(self, Q):
        maze = self.maze
        maze[(3,3)] = -1
        actions = dict()
        for player_i in range(self.maze.shape[0]):
            for player_j in range(self.maze.shape[1]):
                player=(player_i,player_j)
                if(player!=(3,3)):
                    s = (player,(3,3))
                    s_index = self.map[s]
                    actions[player] = np.argmax(Q[s_index, :])
        self.draw_maze(maze, actions)

def getLearningRate(n):
    return 1/((n+1)**(2/3))

def q_learning(env, lambd, eps, s, n_iter):
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    n = np.zeros((n_states, n_actions)) 
    init_s_index = env.map[s]
    s = init_s_index
    V_t = [] 
    for t in range(n_iter):
        V_t.append(np.max(Q[init_s_index, :]))
        a = env.epsilonGreedy(s, eps, Q)
        s_next_index, curr_reward = env.move(s, a)
        lr = getLearningRate(n[s, a])
        best_next_Q = np.max(Q[s_next_index, :])
        Q[s, a] = (1-lr)*Q[s, a] + lr*(curr_reward + lambd*best_next_Q)
        n[s, a] += 1
        s = s_next_index
    return Q, V_t

def sarsa(env, lambd, eps, s, n_iter):
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    n = np.zeros((n_states, n_actions), dtype=int) 
    init_s_index = env.map[s]
    s = init_s_index
    V_t = np.zeros((n_iter,))
    a = env.epsilonGreedy(init_s_index, eps, Q)

    for t in range(n_iter):
        V_t[t] = np.max(Q[init_s_index, :])
        s_next_index, curr_reward = env.move(s, a)
        a_next = env.epsilonGreedy(s_next_index, eps, Q)
        lr = getLearningRate(n[s, a])
        Q[s, a] = (1-lr)*Q[s, a] + lr*(curr_reward + lambd*Q[s_next_index, a_next])
        n[s, a] += 1
        s = s_next_index
        a = a_next
    return Q, V_t



