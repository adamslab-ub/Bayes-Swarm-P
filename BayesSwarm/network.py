#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np 

from BayesSwarm.util import get_distance_from_line, get_line_equation, check_point_is_between_two_points
from BayesSwarm.util import tic, toc


class Network:
    def __init__(self, n_robots, is_full_observation=True, communication_range=20): #46 meters (150 feet)
        self.is_full_observation = is_full_observation
        self.communication_range = communication_range
        self.n_robots = n_robots
        self.robots_data_packets = {}
        self.neighbours_list = []
        for i in range(n_robots):
            robot_name = "robot-"+str(i)
            self.robots_data_packets[robot_name] = {}
            self.robots_data_packets[robot_name]["timestamp"] = -1. 
            self.robots_data_packets[robot_name]["plan"] = []
            self.robots_data_packets[robot_name]["observations"] = []
            self.robots_data_packets[robot_name]["belief_model"] = {}
            self.robots_data_packets[robot_name]["known_fake_source"] = [] 
            
            self.neighbours_list.append(robot_name)
        self.broadcasted_information = {}

    def broadcast_information(self, t, robot_name, data_packet):
        self.robots_data_packets[robot_name]["timestamp"] = t
        self.robots_data_packets[robot_name]["plan"] = data_packet["plan"]
        self.robots_data_packets[robot_name]["observations"] = data_packet["observations"]
        self.robots_data_packets[robot_name]["belief_model"] = data_packet["belief_model"]
        self.robots_data_packets[robot_name]["known_fake_source"] = data_packet["known_fake_source"]
        
        
        self.broadcasted_information = {"robot_name": robot_name, "timestamp": t, "belief_model": data_packet["belief_model"],\
                                        "plan": data_packet["plan"], "observations": data_packet["observations"],\
                                        "known_fake_source": data_packet["known_fake_source"]}

    def get_information(self):
        return self.broadcasted_information
        
    def get_cached_information(self):
        return self.robots_data_packets

    def get_neighbours_list(self, robot_name, robots_location=None):
        if self.is_full_observation:
            neighbours_list = list(np.copy(self.neighbours_list))
            neighbours_list.remove(robot_name)
        else:
            neighbours_list = self.find_neighbours(robot_name, robots_location)
        return neighbours_list

    def find_neighbours(self, robot_name, robots_location=None):
        if robots_location == None:
            raise('The location of robots (robots_location) is not provided!')

        neighbours_list_robot_name = list(np.copy(self.neighbours_list))
        peers_list_robot_name = list(np.copy(self.neighbours_list))
        neighbours_list_robot_name.remove(robot_name)
        peers_list_robot_name.remove(robot_name)
        robot_location = robots_location[robot_name]
        for neighbour in peers_list_robot_name:
            if np.linalg.norm(robots_location[neighbour] - robot_location) > self.communication_range:
                neighbours_list_robot_name.remove(neighbour)

        return neighbours_list_robot_name
