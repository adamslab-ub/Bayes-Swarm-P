#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/27/2020 """

import numpy as np 
from BayesSwarm.simulator import Simulator

import scipy.io


def main():
    debug = False
    case_study = 76
    decision_making_mode = "bayes-swarm" # "bayes-swarm"
    
    is_scout_team = False
    start_locations = None
    observation_frequency = 1
    time_profiling_enable = False
    optimizers=[None, None]

    if case_study == 1:
        debug = True
        n_robots = 4
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 71:
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
        time_profiling_enable = False
    elif case_study == 72:
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = False #True
        bayes_swarm_mode = "local-penalty"
        time_profiling_enable = False
    elif case_study == 73:
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty-sync"
        time_profiling_enable = False
    elif case_study == 74:
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "base"
        time_profiling_enable = False
    elif case_study == 75:
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "explorative-penalized"
        time_profiling_enable = False
    elif case_study == 76: # Levy-walk method
        n_robots = 5
        source_id = 7
        decision_making_mode = "levy-walk" # "bayes-swarm"
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = None
        time_profiling_enable = False
    elif case_study == 11:
        n_robots = 5
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 12:
        n_robots = 5
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = False #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 13:
        n_robots = 5
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty-sync"
    elif case_study == 14:
        n_robots = 5
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "base"
    elif case_study == 15:
        n_robots = 5
        source_id = 1
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "explorative-penalized"
    elif case_study == 21:
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 22:
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = False #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 23:
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty-sync"
    elif case_study == 24:
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "base"
    elif case_study == 25:
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "explorative-penalized"
    elif case_study == 26:
        n_robots = 10
        source_id = 2
        decision_making_mode = "levy-walk" # "bayes-swarm"
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = None
        time_profiling_enable = False
    elif case_study == 3:
        #optimizers=["PSO", None]
        n_robots = 5
        source_id = 3
        filtering_mode = "none"
        enable_full_observation = True
        bayes_swarm_mode = "local-penalty" #extended-local-penalty"
        #lb = np.array([-3, -3])
        #ub = np.array([-1.2, 3])
        #start_locations = np.random.rand(n_robots,2) * (ub - lb) + lb
        scale = 1
        mat = scipy.io.loadmat('data/GSOCase_InitLoc.mat')
        start_locations = mat["startLoc"][:n_robots,:]*scale
        observation_frequency = 10
    elif case_study == 5:
        n_robots = 10
        source_id = 5
        filtering_mode = "none"
        enable_full_observation = True
        bayes_swarm_mode = "local-penalty"
        #lb = np.array([-3, -3])
        #ub = np.array([-1.2, 3])
        #start_locations = np.random.rand(n_robots,2) * (ub - lb) + lb
        start_locations = np.ones((n_robots,2))*[-70,-70]
        observation_frequency = 1
    elif case_study == 6:
        n_robots = 5
        source_id = 5
        filtering_mode = "none"
        enable_full_observation = True
        bayes_swarm_mode = "local-penalty"
        start_locations = np.ones((n_robots,2))*[-70,-70]
        observation_frequency = 1
    elif case_study == 210: # Time profiling
        time_profiling_enable = True
        n_robots = 10
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 710: # Time profiling of Case 7
        time_profiling_enable = True
        n_robots = 5
        source_id = 7
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 810: # Time profiling of Case 8
        time_profiling_enable = True
        n_robots = 5
        source_id = 8
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 811: # Time profiling of Case 8 with large team
        time_profiling_enable = True
        n_robots = 30
        source_id = 8
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 91: # Analysis of main optimizer
        time_profiling_enable = True
        n_robots = 5
        source_id = 9
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 81:
        n_robots = 5
        source_id = 8
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    elif case_study == 110: # Study the impact of penalty term
        n_robots = 2
        source_id = 2
        filtering_mode = "none" #"none"
        enable_full_observation = True #True
        bayes_swarm_mode = "local-penalty"
    else:
        raise("Invalid Case Study!")

    sim = Simulator(n_robots=n_robots, source_id=source_id, start_locations=start_locations,\
                    decision_making_mode=decision_making_mode, bayes_swarm_mode=bayes_swarm_mode,\
                    filtering_mode=filtering_mode, observation_frequency=observation_frequency,\
                    optimizers=optimizers, enable_full_observation=enable_full_observation,\
                    is_scout_team=is_scout_team, debug=debug, time_profiling_enable=time_profiling_enable)
    sim.run()
    sim.get_mission_metrics()

if __name__ == "__main__":
    # execute only if run as a script
    main()