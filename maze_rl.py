import numpy as np
import animation_util as animation

POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']


class RL_Maze():
    def __init__(self, maze_list, start=(1, 0), goal=(14, 15), step_cost=0):
        self.grid = np.array(maze_list)
        self.start = start
        self.goal = goal
        self.step_cost = step_cost

        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]

        self.i = start[0]
        self.j = start[1]
        self.curr_state = start
        # 1 (True) is a wall, 0 (False) is a path
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1

        # start and goal need to be free cells
        self.grid[start] = 0
        self.grid[goal] = 0

    def setup(self):

        # keep track of visited cells
        self.visited = set()
        self.recorded_path = list()
        self.move_n = 0
        self.free_cells = {(r, c) for r in range(self.height)
                           for c in range(self.width) if self.grid[r, c] == 0
                           }
        # self.free_cells.remove(self.goal)

        # rewards should be a dict of: (i, j): r
        # such that (S) : reward
        # actions should be a dict of: (i, j): A[]
        # such that (S): list of possible actions
        actions = {}

        for i in self.free_cells:
            actions[i] = self.get_valid_actions(i)

        self.actions = actions

    def clear_path(self):
        self.recorded_path = list()

    def set_state(self, s):
        self.i, self.j = s
        self.curr_state = s

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys())

    def game_over(self):
        # returns true if game is over, else false
        for s in self.free_cells:
            if s not in self.visited:
                return False
        return True

    def get_reward(self, state):
        if self.game_over():
            return 5
        elif state in self.visited:
            return -3 + self.step_cost
        else:
            return 2 + self.step_cost

    # return valid actions of current state
    def get_valid_actions(self, state=None):
        if state == None:
            state = self.curr_state

        nrows, ncols = self.grid.shape
        grid = self.grid

        actions = {(a) for a in POSSIBLE_ACTIONS}
        x, y = state

        if x == 0 or grid[x - 1, y] != 0:
            actions.remove('U')
        if x == nrows-1 or grid[x + 1, y] != 0:
            actions.remove('D')
        if y == 0 or grid[x, y - 1] != 0:
            actions.remove('L')
        if y == ncols-1 or grid[x, y + 1] != 0:
            actions.remove('R')

        return actions  # np.array

    def get_next_state_reward(self, action, s=None):
        if s == None:
            s = self.curr_state
        i, j = s
        if action in self.actions[s]:
            if action == 'U':
                i -= 1
            elif action == 'D':
                i += 1
            elif action == 'R':
                j += 1
            elif action == 'L':
                j -= 1
        # return a reward (if any)
        s_prime = (i, j)
        reward = self.get_reward(s_prime)
        return ((i, j), reward)  # (s', r)

    def move(self, action):
        valid_actions = self.get_valid_actions()
        if action in valid_actions:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
            self.visited.add(self.curr_state)

            self.recorded_path.append(self.curr_state)
            self.move_n += 1
            self.curr_state = (self.i, self.j)


"""
if __name__ == '__main__':
    rl_maze = RL_Maze(maze_list, (1, 0), (14, 15))
    animation.show_maze(rl_maze)

    # animation.animation(rl_maze, 20)
"""
