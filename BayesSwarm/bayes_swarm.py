#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np
from math import erfc, erf
from scipy.optimize import minimize
from scipy.spatial import distance
import pickle
from scipy.stats import norm

from pyswarm import pso


from BayesSwarm.gp_modeling import GpModeling
from BayesSwarm.gpy_modeling import GpyModeling
from BayesSwarm.util import *


class BayesSwarm:
    def __init__(self, robot, source, time_max, local_penalizing_coef,
                 bayes_swarm_mode="local-penalty", alpha_mode="adaptive",
                 decision_horizon_mode="constant", alpha=0.1, beta=1,
                 optimizers=["COBYLA","L-BFGS-B"], debug=False, time_profiling_enable=False,
                 depot_mode="single-depot"):
        self.time_profiling_enable = time_profiling_enable
        if self.time_profiling_enable:
            self.time_profiling = {"data_size": [], "belief_update": [], "cal_expected_target": [],
                                    "cal_next_location": [], "overal_time": [], "mission_time": []} 
        self.depot_mode = depot_mode
        self.debug = debug
        self.time_max = time_max
        self.safe_distance = 0.1
        self.min_distance_to_fake_source = 100
        self.beta = beta
        self.min_dist_max = 100
        self.min_distance_to_xbest = self.min_dist_max
        self.is_non_batch_mode = True

        # Having noise help the stability of the GPR training (avoid zero covariance)
        self.signal_noise_in_modeling = 1e-3 # std-dev
        
        self.robot = robot
        self.source = source
        self.robot_id = self.robot.get_robot_id()

        ## Beign: Solver Analysis
        ## robot_id_for_analysis = 2
        self.time_threshold = 0
        self.main_optimizer_analysis_enable = False
        #optimizer = "PSO"
        #if self.robot_id == robot_id_for_analysis:
        #    self.main_optimizer_analysis_enable = True
        self.mu_optimizer_analysis_enable = False
        #if self.robot_id == robot_id_for_analysis:
        #    self.mu_optimizer_analysis_enable = True
        ## End: Solver Analysis
        
        if bayes_swarm_mode == "extended-local-penalty":
            self.is_robotic_local_penalizing = True
        else:
            self.is_robotic_local_penalizing = False
        self.is_hard_local_penalizer = False
        self.is_stagnation_penalizing = False
        self.is_estimated_L = False
        self.is_estimated_M = True
        self.is_knowledge_per_meter = False
        self.bayes_swarm_mode = bayes_swarm_mode
        self.alpha = alpha
        self.alpha_mode = alpha_mode
        self.decision_horizon_mode = decision_horizon_mode 
        self.local_penalizing_coef = local_penalizing_coef
        self.Omega_coeff = 1
        self.Omega = -1.
        self.Sigma = -1.
        self.Gamma = -1.
        self.jitter = 1e-3
        self.location_current, self.robot_heading = robot.get_robot_position()
        velocity, decision_horizon, decision_horizon_init, source_detection_range = source.get_source_info_robot()
        self.angular_range, self.arena_lb, self.arena_ub = source.get_source_info_arena()
        self.robot_velocity = velocity
        self.decision_horizon = decision_horizon
        self.decision_horizon_init = decision_horizon_init
        if optimizers[0] == None:
            self.optimizer = "COBYLA"
        else:
            self.optimizer = optimizers[0]

        if optimizers[1] == None:
            self.mu_optimizer = "L-BFGS-B"
        else:
            self.mu_optimizer = optimizers[1]
        self.search_dim = 2 # 2D Space
        self.expected_source_location = []
        self.expected_source_location_prv = []
        self.expected_source_magnitude = -1
        self.expected_source_sigma = -1
        self.next_point = []
        self.X_peers_plan = []
        self.known_fake_source = [] 
        
        self.covered_lb = self.location_current
        self.covered_ub = self.location_current

        signal_noise_in_modeling, self.observation_frequency = self.robot.get_sensor_info()
        if signal_noise_in_modeling > 0:
            self.signal_noise_in_modeling = signal_noise_in_modeling
        self.gp_modeling_mode = "scipy-gpr" #"gpy" # "scipy-gpr"
        if self.gp_modeling_mode == "scipy-gpr":
            self.gp_mu = GpModeling(self.signal_noise_in_modeling, optimizer='fmin_l_bfgs_b')
            self.gp_mu_extended = GpModeling(self.signal_noise_in_modeling, optimizer='fmin_l_bfgs_b')
            self.gp_sigma = GpModeling(self.signal_noise_in_modeling, optimizer='fmin_l_bfgs_b')
        else: #"gpy"
            self.gp_mu = GpyModeling(self.signal_noise_in_modeling)
            self.gp_mu_extended = GpyModeling(self.signal_noise_in_modeling)
            self.gp_sigma = GpyModeling(self.signal_noise_in_modeling)

        self.decision_counter = 0
        self.robot_movement_min = self.robot_velocity / self.observation_frequency
        if self.decision_horizon * self.observation_frequency < 1:
            raise("Increase decision horizon or observation frequency; or decrease the robot velocity!")
        self.displacement_threshold_min = source_detection_range * 0.9

        
        self.is_robot_stuck_in_local = False
        self.is_sync_decision_making = False

    def bayes_swarm_jcise2019(self): # "JCISE-2019"
        self.bayes_swarm_mode = "base"
        self.alpha_mode = "constant"
        
    def bayes_swarm_mrs2019(self): # "MRS-2019"
        self.bayes_swarm_mode = "local-penalty"
        self.alpha_mode = "adaptive"

    def bayes_swarm_iros2020(self): # "IROS-2020"
        self.bayes_swarm_mode = "scalable"
        self.alpha_mode = "adaptive"

    def get_next_point(self, t, X_robot, y_robot):        
        if self.time_profiling_enable:
            tic()
        # Update the exploitation weight (alpha) 
        self.location_current, self.robot_heading = self.robot.get_robot_position()
        self.update_decision_horizon(t)
        if self.decision_counter == 0 or np.size(y_robot) <= 1:
            self.next_point, self.next_point_magnitude = self.get_first_decision(self.depot_mode)
            if self.time_profiling_enable:
                self.time_profiling["belief_update"].append(0)
                self.time_profiling["cal_expected_target"].append(0)
                self.time_profiling["cal_next_location"].append(0)
            result = None
        else:
            self.X_peers_plan = self.robot.get_peers_plan()
            n_peers = len(self.X_peers_plan)
            X_peers = []
            X_peers_next_waypoint = []
            
            if n_peers > 0:
                i = 0
                for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        if i == 0:
                            X_peers_next_waypoint = x_peer_plan[2:]
                            X_peers_dummy = self.get_interpoints(x_peer_plan[:2], x_peer_plan[2:])
                            if np.size(X_peers_dummy) > 0:
                                X_peers = X_peers_dummy
                                i += 1
                        else:
                            X_peers_dummy = self.get_interpoints(x_peer_plan[:2], x_peer_plan[2:])
                            if np.size(X_peers_dummy) > 0:
                                X_peers = np.vstack((X_peers, X_peers_dummy))
                            X_peers_next_waypoint = np.vstack((X_peers_next_waypoint, x_peer_plan[2:]))
 
            if self.time_profiling_enable:
                tic()
            # 1. Update Belief & Knowledge Uncertainty models
            self.update_model(X_robot, y_robot, X_peers)

            if self.time_profiling_enable:
                dummy_time = toc()
                self.time_profiling["belief_update"].append(dummy_time)
                tic()
            
            # 2. Find the expected location of the source signal
            self.compute_expected_source_location()
            self.update_min_distance_to_xbest(X_peers_next_waypoint)
            self.update_exploitation_weight(t)

            if self.time_profiling_enable:
                dummy_time = toc()
                self.time_profiling["cal_expected_target"].append(dummy_time)
                tic()
            
            # 3. Find the next point
            x, f, _ = self.constraint_optimizer(objective_function=self.acquisition_function,
                                                optimizer=self.optimizer, lb=self.arena_lb, ub=self.arena_ub,
                                                constraints=self.motion_constrained, x0=self.location_current)
            self.next_point = x
            self.next_point_magnitude = f

            if self.main_optimizer_analysis_enable and t >= self.time_threshold:
                optimizer_analysis = {"PSO": [], "COBYLA": [], "SLSQP": [], "trust-constr": [],
                                       "current-location": self.location_current, "Acquisition-Func": []}
                optimizer_list = ["PSO", "COBYLA", "SLSQP", "trust-constr"]
                #x0 = np.random.rand() * (self.source.arena_ub - self.source.arena_lb) + self.source.arena_lb
                x0 = self.location_current
                bounds = [(self.arena_lb[0], self.arena_ub[0])] 
                n_dim = np.size(self.arena_lb)
                for i in range(1, n_dim):
                    bounds += [(self.arena_lb[i], self.arena_ub[i])]
                bounds = tuple(bounds)
                for optimizer in optimizer_list:
                    self.optimizer = optimizer
                    print(optimizer)
                    tic()
                    x, f, result = self.constraint_optimizer(objective_function=self.acquisition_function,
                                    optimizer=self.optimizer, lb=self.arena_lb, ub=self.arena_ub,
                                    constraints=self.motion_constrained, x0=self.location_current)
                    self.next_point = x
                    self.next_point_magnitude = f
                    computing_time = toc()
                    results = {"computing_time": computing_time, "optimization_result": result}
                    optimizer_analysis[optimizer] = results
                
                with open('results_optimizer_analysis.pickle', 'wb') as handle:
                    pickle.dump(optimizer_analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(optimizer_analysis)
                quit() 
            
            if self.time_profiling_enable:
                dummy_time = toc()
                self.time_profiling["cal_next_location"].append(dummy_time)
                
            if self.is_robot_stuck_in_local == True:
                #dx = self.robot_velocity * self.decision_horizon
                dx = self.expected_source_location - self.location_current
                if np.linalg.norm(dx) < self.displacement_threshold_min:                    
                    x, f, _ = self.constraint_optimizer(objective_function=self.acquisition_function_at_local_optima,
                                                        optimizer=self.optimizer, lb=self.arena_lb, ub=self.arena_ub, 
                                                        constraints=self.motion_constrained, x0=self.location_current)
                    self.next_point = x
                    self.next_point_magnitude = f
                    self.set_local_source(self.expected_source_location)
                else:                    
                    V = self.robot_velocity
                    T = self.decision_horizon
                    dx_norm = V * T
                    theta = np.random.rand() * np.pi * 2
                    dx_dir = np.array([np.cos(theta), np.sin(theta)])
                    dx_move =  dx_norm * np.clip(np.random.rand(), 0.1, 0.9) * dx_dir
                    self.next_point =  self.location_current + dx_move
                    self.next_point = np.clip(self.next_point, self.arena_lb, self.arena_ub)
                    self.next_point_magnitude = self.gp_mu.predict(self.next_point)
                self.is_robot_stuck_in_local = False
                
            if self.debug == True:
                print("=== Acquisition Function ====")
                print("xbest: ", self.expected_source_location)
                print("Current Location: ", self.location_current)
                print("Selected: ", self.next_point)
                print("Omega: ", self.Omega)
                print("Sigma: ", self.Sigma)
                print("Gamma: ", self.Gamma)
                print("Alpha: ", self.alpha)
                print("Reward: ", self.reward_value)
        self.decision_counter += 1
        if self.time_profiling_enable:
            dummy_time = toc()
            self.time_profiling["overal_time"].append(dummy_time)
            self.time_profiling["data_size"].append(len(X_robot))
            self.time_profiling["mission_time"].append(t)

        # Clip locations to avoid crossing the boarders
        self.next_point = np.clip(self.next_point, self.arena_lb, self.arena_ub)
        self.next_point_magnitude = self.gp_mu.predict(self.next_point)
        if self.debug:
            print(result)
            print(self.location_current, self.next_point)
            print(self.arena_lb, self.arena_ub)
            print(self.motion_constrained(self.next_point), np.linalg.norm(self.next_point-self.location_current))

        return self.next_point  

    def constraint_optimizer(self, objective_function, optimizer, lb, ub, constraints, x0=None):
        " Constraint minimizers "
        if x0.any() == None:
            x0 = np.random.rand() * (ub - lb) + lb
        if optimizer == "SLSQP":
            n_dim = np.size(lb)
            bounds = [(lb[0], ub[0])] 
            for i in range(1, n_dim):
                bounds += [(lb[i], ub[i])]
            bounds = tuple(bounds)
            cons = ({'type': 'ineq', 'fun': lambda x:  constraints(x)})
            result = minimize(objective_function, x0.reshape(-1,), method=optimizer, bounds=bounds, constraints=cons)
            next_point = result.x
            next_point_magnitude = result.fun
        elif optimizer == "COBYLA":
            n_dim = np.size(lb)
            bounds = [(lb[0], ub[0])] 
            for i in range(1, n_dim):
                bounds += [(lb[i], ub[i])]
            bounds = tuple(bounds)
            cons = []
            cons.append({'type': 'ineq', 'fun': lambda x:  constraints(x)})
            for factor in range(len(bounds)):
                lower, upper = bounds[factor]
                l = {'type': 'ineq',
                    'fun': lambda x, lb=lower, i=factor: x[i] - lb}
                u = {'type': 'ineq',
                    'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                cons.append(l)
                cons.append(u)
            result = minimize(objective_function, x0.reshape(-1,), method=self.optimizer, constraints=cons)
            next_point = result.x
            next_point_magnitude = result.fun
        elif optimizer == "trust-constr":
            n_dim = np.size(lb)
            bounds = [(lb[0], ub[0])] 
            for i in range(1, n_dim):
                bounds += [(lb[i], ub[i])]
            bounds = tuple(bounds)
            cons = ({'type': 'ineq', 'fun': lambda x:  constraints(x)})
            result = minimize(objective_function, x0.reshape(-1,), method=self.optimizer, bounds=bounds, constraints=cons)
            next_point = result.x
            next_point_magnitude = result.fun
        else: # "PSO"                
            result = pso(objective_function, lb, ub, f_ieqcons=constraints,\
                        maxiter=100, swarmsize=100, debug=False)
            next_point = result[0]
            next_point_magnitude = result[1]

        return next_point, next_point_magnitude, result
            
    def save_time_profiling(self, file_name):
        time_profiling = self.time_profiling
        with open(file_name+'_time_profiling.pickle', 'wb') as handle:
            pickle.dump(time_profiling, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_covered_area(self, covered_lb, covered_ub):
        self.covered_lb = covered_lb
        self.covered_ub = covered_ub
        
    def get_first_decision(self, depot_mode="single-depot"):
        """
        Args:
            depot_mode (str): Starting point mode: "single-depot", "four-depot" 
        """
        robot_id = self.robot_id
        n_robots = self.robot.get_n_robots()
        travel_dist_max = self.robot_velocity * self.decision_horizon_init
        
        theta_sign = [1, 1]
        if depot_mode == "four-depot":
            d_theta = np.pi / 2
            n_robot_per_corner = n_robots // 4
            if robot_id < n_robot_per_corner: # Lower left corner
                robot_id_at_corner = robot_id
                theta_sign = [1, 1]
            elif (n_robot_per_corner <= robot_id
                    and robot_id < 2 * n_robot_per_corner): # Lower right corner
                robot_id_at_corner = robot_id - n_robot_per_corner
                theta_sign = [-1, 1]
            elif (2 * n_robot_per_corner <= robot_id
                    and robot_id < 3 * n_robot_per_corner): # Upper right corner
                robot_id_at_corner = robot_id - 2 * n_robot_per_corner
                theta_sign = [-1, -1]
            else: # Upper left corner
                robot_id_at_corner = robot_id - 3 * n_robot_per_corner
                n_robot_per_corner = n_robots - 3 * n_robot_per_corner
                theta_sign = [1, -1]
            if np.abs(d_theta - 2 * np.pi) < 1e-3:
                theta = (robot_id_at_corner + 1) * d_theta / n_robot_per_corner
            else:
                theta = (robot_id_at_corner + 1) * d_theta / (n_robot_per_corner + 1)
        else:
            d_theta = self.angular_range[1] - self.angular_range[0]
            if np.abs(d_theta - 2 * np.pi) < 1e-3:
                theta = (robot_id+1) * d_theta / n_robots
            else:
                theta = (robot_id+1) * d_theta / (n_robots + 1)
        next_direction = np.array([theta_sign[0] * np.cos(theta), theta_sign[1] * np.sin(theta)])
        dummy_next_point = self.location_current + next_direction * travel_dist_max
        next_point = np.clip(dummy_next_point, self.arena_lb, self.arena_ub)
        next_point_magnitude = np.linalg.norm(next_point) #np.random.rand()

        return next_point, next_point_magnitude

    def update_model(self, X_robot, y_robot, X_peers):
        ## Update the Belief model (the expected value of the source signal)
        self.gp_mu.update(X_robot,y_robot)

        ## Update the Knowledge Uncertainty model (the uncertainity value of the source signal)
        # Compute the extended input (X_extended)
        if not X_peers == []:
            y_peers, _ = self.gp_mu.predict(X_peers)
            X_extended = np.vstack((X_robot, X_peers))
            y_extended = np.hstack((y_robot, y_peers))
            #self.gp_mu.get_hyperparameters()
            #self.gp_sigma.set_hyperparameters(kernel=)
            self.gp_sigma.update(X_extended, y_extended)
        else:
            self.gp_sigma = self.gp_mu 

    def set_local_source(self, location):
        if np.size(location) > 0:
            if np.size(self.known_fake_source) > 0:
                dist = np.min(distance.cdist(location.reshape(-1,2), self.known_fake_source.reshape(-1,2)))
                if dist > 0:
                    self.known_fake_source = np.vstack((self.known_fake_source, location))
            else:
                self.known_fake_source = location

    def get_local_source(self):
        return self.known_fake_source

    def acquisition_function_at_local_optima(self, x):
        self.Sigma = self.knowledge_gain(x)
        self.Gamma = 1
        if np.size(self.X_peers_plan) > 0:
            if self.is_robotic_local_penalizing:
                self.Gamma = self.robotic_local_penalizing(x)
            else:
                self.Gamma = self.local_penalizing(x)
        reward_value = self.Sigma * self.Gamma * np.linalg.norm(self.location_current - x)
        return - reward_value


    def acquisition_function(self, x, bayes_swarm_mode=None):
        if bayes_swarm_mode is None:
            bayes_swarm_mode = self.bayes_swarm_mode
        if self.optimizer == "SLSQP" or self.optimizer == "COBYLA" or self.optimizer == "trust-constr":
            x = x.reshape((-1,))

        if bayes_swarm_mode == "pure-exploitative":
            self.Omega = self.source_seeking(x)
            self.Sigma = 0
            self.Gamma = 1
        elif bayes_swarm_mode == "pure-explorative":
            self.Omega = 0
            self.Sigma = self.knowledge_gain(x)
            self.Gamma = 1
        elif bayes_swarm_mode == "base":
            self.Omega = self.source_seeking(x)
            self.Sigma = self.knowledge_gain(x)
            self.Gamma = 1
        elif bayes_swarm_mode == "exploitative-penalized":
            self.Omega = self.source_seeking(x)
            self.Sigma = 0
            self.Gamma = 1
            if np.size(self.X_peers_plan) > 0:
                if self.is_robotic_local_penalizing:
                    self.Gamma = self.robotic_local_penalizing(x)
                else:
                    self.Gamma = self.local_penalizing(x)
        elif bayes_swarm_mode == "explorative-penalized":
            self.Omega = 0
            self.Sigma = self.knowledge_gain(x)
            self.Gamma = 1
            if np.size(self.X_peers_plan) > 0:
                if self.is_robotic_local_penalizing:
                    self.Gamma = self.robotic_local_penalizing(x)
                else:
                    self.Gamma = self.local_penalizing(x)
        elif bayes_swarm_mode == "pure-exploitative-self-interest":
            self.Omega = 1 #self.source_seeking(x)
            self.Sigma = self.diffrence_score(x)
            self.Gamma = 1
        elif bayes_swarm_mode == "local-penalty-sync":
            self.is_sync_decision_making = True
            self.Omega = self.source_seeking(x)
            self.Sigma = self.knowledge_gain(x)
            self.Gamma = 1
            if np.size(self.X_peers_plan) > 0:
                if self.is_robotic_local_penalizing:
                    self.Gamma = self.robotic_local_penalizing(x)
                else:
                    self.Gamma = self.local_penalizing(x)
        else: #"local-penalty"
            self.Omega = self.source_seeking(x)
            self.Sigma = self.knowledge_gain(x)
            self.Gamma = 1
            if np.size(self.X_peers_plan) > 0:
                if self.is_robotic_local_penalizing:
                    min_dist = self.min_distance_to_fake_source
                    self.Gamma = self.robotic_local_penalizing(x)# * (1 - np.exp(-10*(min_dist/(np.sqrt(2)*4*self.safe_distance))**2))
                    
                else:
                    self.Gamma = self.local_penalizing(x)
        if self.is_stagnation_penalizing:
            self.Gamma *= self.stagnation_penalizing(x)

        # if self.debug == True:
        #     print("=== Acquisition Function ====")
        #     print("xbest: ", self.expected_source_location)
        #     print("Current Location: ", self.location_current)
        #     print("Selected: ", x)
        #     print("Omega: ", self.Omega)
        #     print("Sigma: ", self.Sigma)
        #     print("Gamma: ", self.Gamma)
        #     print("Alpha: ", self.alpha)
        #     print("Reward: ", self.reward_value)
        self.Sigma = self.beta * self.Sigma
        if self.is_robotic_local_penalizing:
            if self.is_non_batch_mode:
                self.reward_value = (self.alpha * self.Omega + (1-self.alpha) * self.beta * self.Sigma) * self.Gamma #(1-self.alpha) * 
            else:
                self.reward_value = (self.alpha * self.Omega + (1-self.alpha) * self.beta * self.Sigma) * self.Gamma
        else:
            self.reward_value = (self.alpha * self.Omega + (1-self.alpha) * self.beta * self.Sigma) * self.Gamma

        # Clip value to a large value to avoid numerical issue in the solver
        #inf_val = 1e4
        #self.reward_value = np.clip(self.reward_value, -inf_val, inf_val)
        return -self.reward_value

    def motion_constrained(self, x):
        # For PSO (pyswarm) & Scipy, Format: x + y + c >= 0
        if not self.optimizer == "PSO":
            x = x.reshape((-1,))
            
        V = self.robot_velocity
        T = self.decision_horizon
        dx = x - self.location_current
        
        if self.is_sync_decision_making:
            fval = -np.abs(V * T - np.linalg.norm(dx))
        else:
            fval = V * T - np.linalg.norm(dx)

        return fval

    def motion_constrained_local(self, x):
        # Format: x + y + c >= 0
        V = self.robot_velocity
        T = self.decision_horizon
        dx = x - self.location_current
        
        if self.is_sync_decision_making:
            fval = -np.abs(V * T - np.linalg.norm(dx))
        else:
            fval = V * T - np.linalg.norm(dx)
        in_0 = min(x[0] - self.covered_lb[0], 0) + min(self.covered_ub[0] - x[0],0)
        in_1 = min(x[1] - self.covered_lb[1], 0) + min(self.covered_ub[1] - x[1],0)
        if np.abs(in_0) + np.abs(in_1) > 0:
            fval = -10
        return fval

    def source_seeking(self, x):
        is_normalized_omega = False
        dx = x - self.expected_source_location
        Omega = self.Omega_coeff / (1+np.dot(dx,dx)) # Compute Omega
        if is_normalized_omega:
            dc = x - self.location_current
            Omega *= np.dot(dc,dc)

        return Omega
    
    def knowledge_gain(self, x):
        self.is_robot_stuck_in_local = False
        
        ## Integral over the path l
        dx = x - self.location_current
        dx_length = np.linalg.norm(dx)

        if dx_length < self.robot_movement_min:
            #print("Robot {} is stuck in a local optimum!".format(self.robot_id))
            Warning("Robot is stuck in a local optimum!")
            Sigma = 0
            self.is_robot_stuck_in_local = True
        else:
            next_sample_locations = self.get_interpoints(self.location_current, x)
            if np.size(next_sample_locations) > 0:
                interpoints = next_sample_locations[1:,:]
                #print(interpoints, ": ", np.shape(interpoints), ": ", np.size(interpoints))
                _, y_sigma = self.gp_sigma.predict(interpoints)

                # Compute Sigma
                Sigma = self.integrate_over_path(y_sigma, dx_length)
                if self.is_knowledge_per_meter:
                    Sigma = Sigma/dx_length
            else:
                Sigma = 0

        return Sigma

    def stagnation_penalizing(self, x):
        n_peers = len(self.X_peers_plan)
        ## Compute Gamma; a product of the local penalty imposed by all peers
        Gamma = 1
        if n_peers > 0:
            for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        xp = x_peer_plan[2:]
                        norm_xxp = np.linalg.norm(x-xp)
                        local_penalty = self.stagnation_penalizing_func(self.safe_distance, norm_xxp)
                        Gamma *= local_penalty
                        #print(M, ": ", local_penalty, ": ", norm_xxp)      
        return Gamma

    def stagnation_penalizing_func(self, safe_distance, norm_xxp):
        #return 0.5 * erfc(-(norm_xxp - 2 * safe_distance)*np.sqrt(np.exp(-sigma_x)/(safe_distance*(1+np.exp(-sigma_x)))))
        return erfc(-(norm_xxp - 2*safe_distance)/safe_distance)


    def robotic_local_penalizing(self, x):
        gp_mu = self.gp_mu
        gp_sigma = self.gp_sigma
        safe_distance = self.safe_distance
        if self.is_non_batch_mode:
            mu_x, sigma_x = gp_sigma.predict(x)
        else:
            mu_x, sigma_x = gp_mu.predict(x)
        # Get peers' next waypoint
        n_peers = len(self.X_peers_plan)
        ## Compute Gamma; a product of the local penalty imposed by all peers
        Gamma = 1
        if n_peers > 0:
            if self.bayes_swarm_mode == "extended-local-penalty":
                for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        #xp_start = x_peer_plan[:2]
                        #xp_end = x_peer_plan[2:]
                        for i in range(2):
                            xp = x_peer_plan[i*2:(i+1)*2]
                            #xp = x_peer_plan[2:]
                            norm_xxp = np.linalg.norm(x-xp)
                            local_penalty = self.robotic_local_penalizing_func(safe_distance, norm_xxp, sigma_x)
                            Gamma *= local_penalty
            else:
                for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        xp = x_peer_plan[2:]
                        norm_xxp = np.linalg.norm(x-xp)
                        local_penalty = self.robotic_local_penalizing_func(safe_distance, norm_xxp, sigma_x)
                        Gamma *= local_penalty
                        #print(M, ": ", local_penalty, ": ", norm_xxp)      
        return Gamma

    def robotic_local_penalizing_func(self, safe_distance, norm_xxp, sigma_x):
        #return 0.5 * erfc(-(norm_xxp - 2 * safe_distance)*np.sqrt(np.exp(-sigma_x)/(safe_distance*(1+np.exp(-sigma_x)))))
        return 0.5 * erfc(-(norm_xxp - safe_distance))

    def local_penalizing_func(self, M, L, norm_xxp, mu_xp, sigma_xp):
        if self.is_hard_local_penalizer:
            Phi = min(L*norm_xxp/(np.abs(mu_xp - M)+1.5*sigma_xp), 1) # Alvi et al. 2019
        else:
            Phi = 0.5 * erfc(-(L*norm_xxp - M + mu_xp)/np.sqrt(2*sigma_xp**2)) # Gonzalez et al. 2016
        return Phi 

    def local_penalizing(self, x):
        gp_mu = self.gp_mu
        gp_sigma = self.gp_sigma
        M = self.local_penalizing_coef["M"]
        L = self.local_penalizing_coef["L"]
        mu_x,_ = gp_mu.predict(x)
        # Get peers' next waypoint
        n_peers = len(self.X_peers_plan)
        ## Compute Gamma; a product of the local penalty imposed by all peers
        Gamma = 1
        if n_peers > 0:
            if self.bayes_swarm_mode == "extended-local-penalty":
                for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        #xp_start = x_peer_plan[:2]
                        #xp_end = x_peer_plan[2:]
                        for i in range(2):
                            xp = x_peer_plan[i*2:(i+1)*2]
                            norm_xxp = np.linalg.norm(x-xp)
                            mu_xp, sigma_xp = gp_mu.predict(xp)
                            # todo: In MRS2019, I used gp_sigma for predicting sigma_r; but that's not a good idea as at xp, 
                            # we have values close to zero for sigma. I need to check to find a better way to handle this. Now I am using gp_mu
                            # as it is more suitable and also it is less computationally expensive.
                            #_, sigma_r = gp_sigma.predict(xp)
                            if self.is_estimated_L:
                                delta_x = 10^-5
                                xp_l = xp + delta_x 
                                mu_xp_l, sigma_xp_l = gp_mu.predict(xp_l)
                                L = np.abs(mu_xp_l - mu_xp) / delta_x
                                #L = 3*np.abs(mu_x - mu_xp) / (norm_xxp + self.jitter) # |DF| <= L|dx|
                                #L = 3*np.abs(mu_x - mu_xp) / (norm_xxp + self.jitter) # |DF| <= L|dx|
                            if sigma_xp == 0:
                                sigma_xp += self.jitter
                             #else:
                            if self.is_estimated_M:
                                M = self.expected_source_magnitude
                            local_penalty = self.local_penalizing_func(M, L, norm_xxp, mu_xp, sigma_xp)
                            Gamma *= local_penalty
            else:
                for i_peer in self.X_peers_plan:
                    x_peer_plan = self.X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        xp = x_peer_plan[2:]
                        norm_xxp = np.linalg.norm(x-xp)
                        mu_xp,sigma_xp = gp_mu.predict(xp)
                        # todo: In MRS2019, I used gp_sigma for predicting sigma_r; but that's not a good idea as at xp, 
                        # we have values close to zero for sigma. I need to check to find a better way to handle this. Now I am using gp_mu
                        # as it is more suitable and also it is less computationally expensive.
                        #_, sigma_r = gp_sigma.predict(xp)
                        if self.is_estimated_L:
                            delta_x = 10^-5
                            xp_l = xp + delta_x 
                            mu_xp_l, sigma_xp_l = gp_mu.predict(xp_l)
                            L = np.abs(mu_xp_l - mu_xp) / delta_x
                            #L = 0.5 #3*np.abs(mu_x - mu_xp) / (norm_xxp + self.jitter) # |DF| <= L|dx|
                        if sigma_xp == 0:
                            sigma_xp += self.jitter
                        if self.is_estimated_M:
                            M = self.expected_source_magnitude
                        local_penalty = self.local_penalizing_func(M, L, norm_xxp, mu_xp, sigma_xp)
                        Gamma *= local_penalty
                        #print(M, ": ", local_penalty, ": ", norm_xxp)      
        return Gamma

    def diffrence_score(self,x):
        gp_mu = self.gp_mu
        gp_sigma = self.gp_sigma
        mu1, sig2 = gp_mu.predict(x)
        mu2, sig1 = gp_sigma.predict(x)
        
        return kl_divergence_norm(x, (mu1,sig1), (mu2, sig2))
        
    def update_decision_horizon(self,t=-1):
        if self.decision_horizon_mode == "adaptive":
            time_max = self.time_max
            self.decision_horizon = 2*self.decision_horizon_init / (1 + np.exp(t/time_max - 1/3))

    def update_min_distance_to_xbest(self, X_peers):
        if np.size(X_peers) > 0 and np.size(self.expected_source_location) > 0:
            if np.size(self.known_fake_source) > 0:
                X_peers = np.vstack((X_peers, self.known_fake_source))
                self.min_distance_to_fake_source = np.min(distance.cdist(self.expected_source_location.reshape(-1,2), self.known_fake_source.reshape(-1,2)))
            self.min_distance_to_xbest = np.min(distance.cdist(self.expected_source_location.reshape(-1,2), X_peers.reshape(-1,2)))
        else:
            self.min_distance_to_xbest = self.min_dist_max


    def update_exploitation_weight(self, t=-1):
        """The coefficient alpha in [0,1] (Eq.  (18))  is  the  exploitation weight
           alpha = 1 would  be  purely  exploitative. In the "adaptive" mode, the swarm  behavior
           is  strongly  explorative  at  the  start  and becomes  increasingly  exploitative  over  waypoint  iterations.

        Args:
            t (float, optional): The current mission (simulation) time. Defaults to -1.
        
        Returns:
            float: The exploitation weight (alpha).
        """
        if self.alpha_mode == 'adaptive':
            time_max = self.time_max * 0.1
            if self.is_non_batch_mode:
                # weight_max = 1
                if self.min_distance_to_fake_source < 2.5 * self.safe_distance:
                    #print(self.known_fake_source)
                    self.alpha = 0.1
                #elif self.min_distance_to_xbest > 0.05*5:
                #    self.alpha = 0.8
                else:
                    # weight = weight_max * np.exp(-10*(self.min_distance_to_xbest/(np.sqrt(2)*self.safe_distance))**2)
                    #print(self.min_distance_to_fake_source, ":", self.min_distance_to_xbest, ": weight ", weight)
                    #self.alpha = (weight_max - weight) #/ (1 + np.exp(-10*(t/time_max - 1/3)))
                    if self.min_distance_to_xbest >= 2 * self.safe_distance:
                        self.alpha = 0.8 #
                    else:
                        self.alpha = 1 / (1 + np.exp(-10*(t/time_max - 1/3)))
                #weight = np.exp(-(self.min_distance_to_xbest/(4*self.safe_distance))**2)
                #self.beta = 1 - self.alpha
            else:
                self.alpha = 1 / (1 + np.exp(-10*(t/time_max - 1/3)))
            #self.alpha = 0.99 * self.close
            #print(self.alpha)
        return self.alpha

    def integrate_over_path(self, fx, dx_length):
        """Approximated the integral using Trapezoid rule.
        
        Args:
            fx (float, matrix): The function values of the intermediate points (robot's belief uncertainity in the search problem)
            dx_length (float): The length of the line we want to take integral over it
        
        Returns:
            float: The approximated integral of the sigma at the "interpoints" locations 
        """
        n_samples = np.size(fx)
        
        intgeral_value = 0 
        for i in range(n_samples):
            y_sigma = fx[i]
            if i == 0 or i == n_samples-1:
                intgeral_value += y_sigma
            else:
                intgeral_value += 2 * y_sigma
         
        intgeral_value = dx_length * intgeral_value / (2*n_samples) #@todo

        return intgeral_value

    def func_optimizer(self, x):
        if self.is_non_batch_mode:
            f_val, _ = self.gp_sigma.predict(x)
        else:
            f_val, _ = self.gp_mu.predict(x)

        return -f_val

    def compute_expected_source_location(self):
        mu_optimizer = self.mu_optimizer
        self.expected_source_location_prv = self.expected_source_location
        func = self.func_optimizer
        if mu_optimizer == "COBYLA":
            n_dim = np.size(self.arena_lb)
            bounds = [(self.arena_lb[0], self.arena_ub[0])] 
            for i in range(1, n_dim):
                bounds += [(self.arena_lb[i], self.arena_ub[i])] 
            bounds = tuple(bounds)
            #x0 = np.random.rand() * (self.arena_ub - self.arena_lb) + self.arena_lb
            x0 = self.location_current
            cons = []
            for factor in range(len(bounds)):
                lower, upper = bounds[factor]
                l = {'type': 'ineq',
                    'fun': lambda x, lb=lower, i=factor: x[i] - lb}
                u = {'type': 'ineq',
                    'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                cons.append(l)
                cons.append(u)
            result = minimize(func, x0.reshape(-1,1), method=mu_optimizer, constraints=cons)
            self.expected_source_location = result.x
            self.expected_source_magnitude = result.fun
            _, self.expected_source_sigma = self.gp_mu.predict(result.x)
        elif mu_optimizer == "L-BFGS-B":
            n_dim = np.size(self.arena_lb)
            bounds = [(self.arena_lb[0], self.arena_ub[0])] 
            for i in range(1, n_dim):
                bounds += [(self.arena_lb[i], self.arena_ub[i])] 
            bounds = tuple(bounds)
            x0 = np.random.rand() * (self.arena_ub - self.arena_lb) + self.arena_lb
            result = minimize(func, x0.reshape(-1,1), method=mu_optimizer, bounds=bounds)
            self.expected_source_location = result.x
            self.expected_source_magnitude = result.fun
            _, self.expected_source_sigma = self.gp_mu.predict(result.x)
        elif mu_optimizer == "Nelder-Mead":
            x0 = np.random.rand() * (self.arena_ub - self.arena_lb) + self.arena_lb
            result = minimize(func, x0.reshape(-1,1), method=mu_optimizer)
            self.expected_source_location = result.x
            self.expected_source_magnitude = result.fun
            _, self.expected_source_sigma = self.gp_mu.predict(result.x0)
        elif mu_optimizer == "TNC":
            n_dim = np.size(self.arena_lb)
            bounds = [(self.arena_lb[0], self.arena_ub[0])] 
            for i in range(1, n_dim):
                bounds += [(self.arena_lb[i], self.arena_ub[i])] 
            bounds = tuple(bounds)
            x0 = np.random.rand() * (self.arena_ub - self.arena_lb) + self.arena_lb
            result = minimize(func, x0.reshape(-1,1), method=mu_optimizer, bounds=bounds)
            self.expected_source_location = result.x
            self.expected_source_magnitude = result.fun
            _, self.expected_source_sigma = self.gp_mu.predict(result.x)
        else: # "PSO"
            if self.mu_optimizer_analysis_enable:
                tic()
            result = pso(func, self.arena_lb, self.arena_ub)
            if self.mu_optimizer_analysis_enable:
                computing_time = toc()
                results = {"computing_time": computing_time, "optimization_result": result}
            self.expected_source_location = result[0]
            self.expected_source_magnitude = result[1]
            _, self.expected_source_sigma = self.gp_mu.predict(result[0])
        
        if self.mu_optimizer_analysis_enable:
            mu_optimizer_analysis = {"PSO": results, "L-BFGS-B": [], "Nelder-Mead": [], "TNC": [], "GP-model": self.gp_mu}
            optimizer_list = ["L-BFGS-B", "Nelder-Mead", "TNC"]
            x0 = np.random.rand() * (self.arena_ub - self.arena_lb) + self.arena_lb
            n_dim = np.size(self.arena_lb)
            bounds = [(self.arena_lb[0], self.arena_ub[0])] 
            for i in range(1, n_dim):
                bounds += [(self.arena_lb[i], self.arena_ub[i])] 
            bounds = tuple(bounds)
            for optimizer in optimizer_list:
                tic()
                if optimizer == "Nelder-Mead":
                    result = minimize(func, x0.reshape(-1,1), method=optimizer)
                else:
                    result = minimize(func, x0.reshape(-1,1), method=optimizer, bounds=bounds)
                computing_time = toc()
                results = {"computing_time": computing_time, "optimization_result": result}
                mu_optimizer_analysis[optimizer] = results

            with open('results_mu_optimizer_analysis.pickle', 'wb') as handle:
                pickle.dump(mu_optimizer_analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(mu_optimizer_analysis)
            quit() 
            
    def get_interpoints(self, x_start, x_end):
        dx = x_end - x_start
        dx_length = np.linalg.norm(dx)
        T = dx_length/self.robot_velocity
        dt = 1/self.observation_frequency
        n_sample = int(np.floor(self.observation_frequency * T))
        if n_sample > 0:
            #dx_step = np.linspace(0,1,n_sample)
            #dx_step = np.vstack((dx_step, dx_step)).reshape(-1,2)
            next_sample_locations = np.zeros((n_sample+1, self.search_dim)) 
            index_list = np.arange(0,n_sample+1)
            for i in range(self.search_dim):
                dummy = index_list * dt * dx[i] / T
                next_sample_locations[:,i] = dummy.reshape(1,-1)
        else:
            next_sample_locations = []
        
        return next_sample_locations

    def get_expected_source(self):

        return self.expected_source_location

    def get_expected_source_info(self):
        expected_source_info = {"xbest": self.expected_source_location,\
                                "mu": self.expected_source_magnitude,\
                                "sigma": self.expected_source_sigma}
        return expected_source_info
    
    def get_penalty_parameters(self):
        L = self.local_penalizing_coef["L"]
        M = self.expected_source_magnitude
        return L, M

    def get_gp_gradient(self, gp, x):
        pass
        #gp_hyperparameters = gp_sigma.get_hyperparameters()

        """     
        def estimate_L(model,bounds,storehistory=True):

        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        
        def df(x,model,x0):
            x = np.atleast_2d(x)
            dmdx,_ = model.predictive_gradients(x)
            res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
            return -res

        samples = samples_multidimensional_uniform(bounds,500)
        samples = np.vstack([samples,model.X])
        pred_samples = df(samples,model,0)
        x0 = samples[np.argmin(pred_samples)]
        res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 200})
        minusL = res.fun[0][0]
        L = -minusL
        if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
        return L
        """
