import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';


class Maze:

    # Actions
    
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4
    STAY       = 0
    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    BANK_REWARD = 10
    CAUGHT_REWARD = -50
    


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment maze.
        """
       
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
        

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
        end = False;
        s = 0;

        for player_i in range(self.maze.shape[0]):
            for player_j in range(self.maze.shape[1]):
                            for police_i in range(self.maze.shape[0]):
                                for police_j in range(self.maze.shape[1]):
                                    police_pos=(police_i,police_j)
                                    player_pos=(player_i,player_j)
                                    temp = (player_pos,police_pos);
                                    states[s] = temp
                                    map[temp] = s;
                                    s += 1;
        return states, map


    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        player_row = self.states[state][0][0]
        player_col = self.states[state][0][1]
        row = player_row + self.actions[action][0];
        col = player_col + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        # minotaur_pos_temp = self.minotaurMove(state)
        if hitting_maze_walls:
            
            player_pos_temp= (player_row,player_col);
        else:
            player_pos_temp=(row,col)
            # print(player_pos_temp)
            # minotaur_pos_temp=(self.states[state][1][0],self.states[state][1][1])
        minotaur_actions = {4: (0, 0), 0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        #stay condition
        possibleNextStates = []
        minotaur_row = self.states[state][1][0]
        minotaur_col = self.states[state][1][1]
        for minotaur_action in minotaur_actions:
            # print(minotaur_actions[minotaur_action])

            row_m = minotaur_row + minotaur_actions[minotaur_action][0];
            col_m = minotaur_col + minotaur_actions[minotaur_action][1];

            hitting_maze_walls_m =  (row_m == -1) or (row_m == self.maze.shape[0]) or \
                        (col_m == -1) or (col_m == self.maze.shape[1]) ;
            if not hitting_maze_walls_m:
                minotaur_pos_temp= (row_m,col_m);
                
                temp = (player_pos_temp,minotaur_pos_temp);
                possibleNextStates.append(self.map[temp])
     
        
        return possibleNextStates

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s,a);
                # print(next_s)
                prob = 1.0 / len(next_s)

                for ns in next_s:
                    transition_probabilities[ns, s, a] = prob
                # transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):

                    next_state = self.__move(s,a);
                    
                    state_reward = 0.0
                    for next_s in next_state:
                        
                        # Rewrd for hitting a wall
                        if self.states[s][0] == self.states[next_s][0] and a != self.STAY:
                            # print('her1')
                            state_reward += self.IMPOSSIBLE_REWARD;
                        if self.states[next_s][0]==self.states[next_s][1]:
                            state_reward += self.IMPOSSIBLE_REWARD;
                        # Reward for reaching the exit
                        elif self.states[s][0] == self.states[next_s][0] and self.maze[self.states[next_s][0]] == 2:
                            state_reward += self.GOAL_REWARD;
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            state_reward += self.STEP_REWARD;
                    rewards[s,a] = state_reward / len(next_state)
                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s][0]]<0:
                        row, col = self.states[next_s][0];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s][0];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();

        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)



def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('maze Heist');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        # for m in range(maze.shape[0]):
        #     for n in range(maze.shape[1]):
        #         grid.get_celld()[(m,n)].set_facecolor(col_map[maze[m,n]])
      
        grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0])].get_text().set_text('Player')

        # dum = grid.get_celld()[(path[i][1])].get_face
        grid.get_celld()[(path[i][1])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path[i][1])].get_text().set_text('Minotuar')

        if i > 0:

            if path[i][0] == path[i-1][0]:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player stays')
            else:
                grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
                grid.get_celld()[(path[i-1][0])].get_text().set_text('')
            if path[i][1] == path[i-1][1]:
                grid.get_celld()[(path[i][1])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][1])].get_text().set_text('Minotuar stays')
            else:
                grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
                grid.get_celld()[(path[i-1][1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(.1)