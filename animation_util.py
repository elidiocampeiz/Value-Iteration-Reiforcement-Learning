from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import animation
import time
# Note: you may need to update your version of future
# sudo pip install -U future


palettes = {
    # RGB vectors in a 0-1 format palatable to matplotlib
    # Once a palette is selected ('maze') the boolean values in the maze are mapped to the appropriate color

    # index 0 = paths = color value of white and index 1 = walls = color value of black
    'maze': np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 0.5], [0.0, 0.0, 1.0]]),
    'current_location':  np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.0]]),
    # index 0 = not here = black and index 1 = here = blue
    'start_location':  np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
    # index 0 = not here = white and  index 1 = here = red
    'goal_location':  np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
    # index 0 = not here = white and  index 1 = here = green
    'visited':  np.array([[0.0, 0.0, 0.0],  [0.0, 1.0, 0.5]])
}


def CreateImage(boolean_array, palette_name):
    # The next statement uses an array as an index to another array.  Gotta love Python!
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial#head-3f4d28139e045a442f78c5218c379af64c2c8c9e
    return palettes[palette_name][boolean_array.astype(int)]


def show_maze(Maze):
    curr_state = Maze.curr_state
    goal = Maze.goal
    visited = Maze.visited

    image = CreateImage(boolean_array=Maze.grid, palette_name='maze')

    for i in visited:
        image[i] = palettes['visited'][1]

    image[curr_state] = palettes['start_location'][1]
    image[goal] = palettes['goal_location'][1]
    pyplot.figure(figsize=(5, 5))
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.imshow(image, interpolation='nearest')
    pyplot.show()

# TODO: FIX THIS FUNCTION


def animate_maze(Maze, path):
    image = CreateImage(boolean_array=Maze.grid, palette_name='maze')
    fig = pyplot.figure(figsize=(5, 5))
    pyplot.xticks([]), pyplot.yticks([])
    imgplot = pyplot.imshow(image, interpolation='nearest')
    goal = Maze.goal
    start = Maze.start

    image[start] = palettes['start_location'][1]
    image[goal] = palettes['goal_location'][1]

    def CreateFrame(frame):

        # goal = Maze.goal
        # start = Maze.start
        # image[start] = palettes['start_location'][1]
        # image[goal] = palettes['goal_location'][1]

        # image[frame] = palettes['current_location'][1]
        image[start] = palettes['start_location'][1]
        image[goal] = palettes['goal_location'][1]
        image[frame] = palettes['current_location'][1]
        imgplot.set_data(image)
        image[frame] = palettes['visited'][1]

        return imgplot

    anim = animation.FuncAnimation(
        fig, CreateFrame, frames=path, repeat=False,  interval=1)
    # anim.save('myAnimatedMaze.mp4')
    image[goal] = palettes['current_location'][1]
    imgplot.set_data(image)
    pyplot.show()


def my_animation(Maze, it, s=2):
    curr_state = Maze.curr_state
    goal = Maze.goal
    visited = Maze.visited

    image = CreateImage(boolean_array=Maze.grid, palette_name='maze')
    for _ in range(it):
        curr_state = Maze.curr_state
        goal = Maze.goal
        visited = Maze.visited
        for i in visited:
            image[i] = palettes['visited'][1]
        image[curr_state] = palettes['start_location'][1]
        image[goal] = palettes['goal_location'][1]
        pyplot.figure(figsize=(5, 5))
        pyplot.xticks([]), pyplot.yticks([])
        img = pyplot.imshow(image, interpolation='nearest')
        pyplot.show()

        actions = Maze.get_valid_actions()
        Maze.move(np.random.choice(actions))
        time.sleep(s)
        pyplot.close()


def print_values(V, Maze):
    g = Maze
    for i in range(g.width):
        str = "------"*Maze.grid.shape[0]
        print(str)
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, Maze):

    for i in range(Maze.width):
        str = "------"*Maze.grid.shape[0]
        print(str)
        for j in range(Maze.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary

    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val
