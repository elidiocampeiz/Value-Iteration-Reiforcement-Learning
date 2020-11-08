from __future__ import print_function, division
import numpy as np
import maze_rl as RL
import animation_util as GUI

from builtins import range


THETA = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']


def best_action_value(Maze, V, s):
    # finds the highest value action (max_a) from state s, returns the action and value
    best_a = None
    best_value = float('-inf')
    # grid.set_state(s)
    # loop through all possible actions to find the best current action
    for a in Maze.get_valid_actions(s):
        expected_v = 0
        expected_r = 0

        state_prime, r = Maze.get_next_state_reward(a, s)

        expected_r += r
        # print(state_prime, "v:", V[state_prime])
        expected_v += V[state_prime]
        v = expected_r + GAMMA * expected_v
        if v > best_value:
            best_value = v
            best_a = a

    return best_a, best_value


def calculate_values(Maze, V=None):
    # initialize V(s)
    if V == None:
        V = {}
    state_set = Maze.all_states()
    # V -> { s : values}
    for s in state_set:
        V[s] = 0
    # V[Maze.goal] = 0
    # repeat until convergence
    # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }

    # ToDO: Test if Changing the loop to iterate while "not Maze.game_over()"
    #        and update "s" as the current state after each step
    #        would fix the training display issue.
    while not grid.game_over():
        # sum is delta in original equations

        a = ''
        sum = 0
        for s in state_set:
            old_v = V[s]
            a, new_v = best_action_value(Maze, V, s)
            Maze.move(a)  # move agent in the maze to simulate navegations
            V[s] = new_v
            sum = max(sum, np.abs(old_v - new_v))

        if sum < THETA:
            grid.set_state(grid.start)
            # break
    grid.set_state(grid.goal)
    return V


def initialize_random_policy(Maze):
    # policy is a lookup table for state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    # P -> {S:A}
    cells = Maze.free_cells

    for s in cells:
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    return policy


def calculate_greedy_policy(Maze, V):
    policy = initialize_random_policy(Maze)
    # find a policy that leads to optimal value function
    for s in policy.keys():
        # grid.set_state(s)
        # loop through all possible actions to find the best current action
        best_a, _ = best_action_value(Maze, V, s)
        # Maze.move(best_a)
        policy[s] = best_a
    policy[Maze.goal] = 'G'
    return policy

# NOTE: To Start the simulation please close the first visualization windown 
if __name__ == '__main__':
    # this grid gives you a reward of -0.1 for every non-terminal state
    # we want to see if this will encourage finding a shorter path to the goal
    maze_list = [
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1],
        [1,  1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1],
        [1,  0,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1],
        [1,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1],
        [1,  0,  1,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1],
        [1,  0,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  1],
        [1,  1,  0,  0,  0,  1,  1,  0,  0,  0,  1,  1,  1,  1,  0,  1],
        [1,  0,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  1],
        [1,  1,  0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1],
        [1,  0,  0,  1,  1,  0,  1,  1,  1,  0,  1,  0,  1,  0,  0,  1],
        [1,  1,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  1,  1],
        [1,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0],
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    ]
    office_list = [
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  2,  2,  2,  0,  0,  2,  2,  2,  0,  0,  0,  0,  0,  1],
        [1,  0,  2,  2,  2,  0,  0,  2,  2,  2,  0,  0,  0,  0,  0,  1],
        [1,  0,  2,  2,  2,  0,  0,  2,  2,  2,  0,  0,  0,  0,  0,  1],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  1],
        [1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  1],
        [1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  1],
        [1,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  3,  3,  0,  0,  0,  4,  4,  4,  0,  0,  0,  0,  0,  0,  1],
        [1,  3,  3,  0,  0,  0,  4,  4,  4,  0,  0,  0,  0,  0,  0,  1],
        [1,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    ]
    # Ask of step_cost
    # it = (int(input("Choose max number iterations")))
    # step_cost = (float(input("Choose step cost: ")))
    step_cost = 0.5
    grid = RL.RL_Maze(office_list, (0, 13), (0, 14), step_cost)
    grid.setup()

    GUI.show_maze(grid)

    # print rewards
    # print("rewards:")
    # GUI.print_values(grid.rewards, grid)

    # calculate accurate values for each square
    V = calculate_values(grid)

    # calculate the optimum policy based on our values
    policy = calculate_greedy_policy(grid, V)

    # our goal here is to verify that we get the same answer as with policy iteration

    print("values:")
    GUI.print_values(V, grid)
    print("policy:")
    GUI.print_policy(policy, grid)
    print(grid.move_n)
    # path = {}
    # grid.set_state(grid.start)
    # grid.clear_path()
    # while not grid.game_over() and grid.curr_state != grid.goal:
    #     s = grid.curr_state
    #     path[s] = ''
    #     grid.move(policy[s])
    # for i in range(it):
    #     grid.move_n = 0
    #     V = calculate_values(grid, V)
    #     policy = calculate_greedy_policy(grid, V)
    #     n = grid.move_n
    #     print(n)
    #     grid.set_state(grid.start)

    print(grid.move_n)
    # path.pop(grid.start, '')
    # GUI.animate_maze(grid, path)
    GUI.animate_maze(grid, grid.recorded_path)


# use cost 0 n_moves = 4032
