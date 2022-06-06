#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.obstacle_generation import generate_random_obstacles
from src.utilities.plotting import Plot


class ClutteredEnv:
    def __init__(self, source):
        lb = [0,2.4] #source.arena_lb
        ub = [0,2.4] #source.arena_ub
        x_init=(lb[0], ub[0])
        x_goal = (lb[1], ub[1])
        X_dimensions = np.array([x_init, x_goal])  # dimensions of Search Space

        # create search space
        X = SearchSpace(X_dimensions)
        self.X = X
        n = 5
        Obstacles = generate_random_obstacles(X, x_init, x_goal, n)

    def get_path(self, x_current, x_goal):
        
        x_c = (x_current, x_current)  # starting location
        x_g = (x_goal[0], x_goal[1])  # goal location

        Q = np.array([(8, 4)])  # length of tree edges
        r = 1  # length of smallest edge to check for intersection with obstacles
        max_samples = 1024  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal
        # create rrt_search
        rrt = RRT(self.X, Q, x_c, x_g, max_samples, r, prc)
        path = rrt.rrt_search()
    
    def plot_path(self):
        # plot
        plot = Plot("rrt_2d_with_random_obstacles")
        plot.plot_tree(X, rrt.trees)
        if path is not None:
            plot.plot_path(X, path)
        plot.plot_obstacles(X, Obstacles)
        plot.plot_start(X, x_init)
        plot.plot_goal(X, x_goal)
        plot.draw(auto_open=True)
