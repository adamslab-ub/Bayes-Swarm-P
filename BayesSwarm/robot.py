#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np 
import pickle

from BayesSwarm.util import get_distance_from_line, get_line_equation, check_point_is_between_two_points
from BayesSwarm.util import tic, toc
from BayesSwarm.util import kl_divergence_norm

from BayesSwarm.source import Source as signal
from BayesSwarm.bayes_swarm import BayesSwarm
from BayesSwarm.filtering import Filtering


class Robot:
    def __init__(self, id, n_robots, bayes_swarm_args, source,
                 velocity=1, start_location=[0,0], source_detection_range=1e-1,
                 observation_frequency=1, measurement_noise_rate=0):
        self.is_enabled_full_log = False
        self.id = id
        self.n_robots = n_robots
        self.velocity = velocity # [m/s]
        self.traveled_distance = 0 # [m]
        self.decision_counter = 0
        self.decision_computing_time = []
        
        self.penalty_analysis_enable = False

        self.location = np.array(start_location)
        self.trajectory_history = np.array(start_location)
        
        self.waypoint_start = self.location
        self.waypoint_end = []
        self.waypoint_history = self.waypoint_start
        self.movement_direction = []
        self.robot_heading = np.array([0,0])
        self.movement_resolution = 1e-4 # [m]
        self.res_movement_time = 0
        self.measurement_noise_rate = measurement_noise_rate 
        self.reached_time_resolution = self.movement_resolution * self.velocity # [sec]
        self.search_space_dimension = len(start_location)
        if observation_frequency > 0:
            self.observation_frequency = observation_frequency # [Hz]
        else:
            raise('Observation frequency must be a positive value!')

        self.movement_line = []
        
        self.source = source
        self.angular_range, self.arena_lb, self.arena_ub = self.source.get_source_info_arena()
        self.arena_bound = self.arena_ub - self.arena_lb
        self.source_location, self.time_max = self.source.get_source_info_mission()

        self.source_detection_range = source_detection_range
        self.is_found_source = False
        self.dist_to_source = float('nan')
        self.data_packets_all = {}
        self.peers_plan = {}
        for i in range(n_robots):
            if i != id:
                self.data_packets_all['robot-'+str(i)] = {}
                self.peers_plan['robot-'+str(i)] = []
        
        sensor_value = self.source.measure(self.location)
        self.observation_last = np.hstack((self.location, sensor_value, self.id)) # Its latest single observation
        self.observation_history = self.observation_last
        self.observation_history_full = self.observation_last
        self.observation_last_waypoint = [] # History of observations from moving to the latest waypoint trip
        self.observation_history_shared = []
        self.observation_history_self = self.observation_last
        
        filtering_mode = bayes_swarm_args["filtering_mode"]
        self.is_enabled_filtering = False
        self.is_filtering_over_received_information = False
        if filtering_mode == "self_observation":
            self.is_enabled_filtering = True
        elif filtering_mode == "shared_observation":
            self.is_filtering_over_received_information = True
        elif filtering_mode == "full":
            self.is_enabled_filtering = True
            self.is_filtering_over_received_information = True

        self.prediction_gap_threshold = 0.01#0.05#0.2
        self.information_gain_threshold = 0.05
        if self.is_enabled_filtering:
            self.filtering = Filtering(self.prediction_gap_threshold)
        
        local_penalizing_coef = bayes_swarm_args["local_penalizing_coef"]
        bayes_swarm_mode = bayes_swarm_args["bayes_swarm_mode"]
        time_profiling_enable = bayes_swarm_args["time_profiling_enable"]
        depot_mode = bayes_swarm_args["depot_mode"]
        self.decision_making_mode = bayes_swarm_args["decision_making_mode"]
        optimizers = bayes_swarm_args["optimizers"]
        self.is_sharing_observation = True
        if bayes_swarm_mode == "pure-exploitative-self-interest":
            self.is_sharing_observation = False
            bayes_swarm_mode = "pure-explorative"

        if self.decision_making_mode == "bayes-swarm":
            self.decision_making = BayesSwarm(self, self.source, self.time_max, local_penalizing_coef,
                                              bayes_swarm_mode, optimizers=optimizers,
                                              time_profiling_enable=time_profiling_enable,
                                              depot_mode=depot_mode)
        
    def step(self, time_to_simulate):
        old_res_movement_time = self.res_movement_time
        ## Move its last incompleted movement to make its observation (taking sample)
        if self.res_movement_time > 0:
            if time_to_simulate >= self.res_movement_time:
                self.motion_model(self.res_movement_time)
                self.observation_model()
                time_to_simulate -= self.res_movement_time
                self.res_movement_time = 0
            else:
                self.motion_model(time_to_simulate)
                self.res_movement_time -= time_to_simulate
                time_to_simulate = 0
                
        elif self.res_movement_time < 0:
            print(self.res_movement_time)
            raise('Invalid residual movement time!')
        
        if time_to_simulate > 0:
            ## Normal operation to find the inter-waypoint for making observations (taking samples)
            n_samples_dummy = time_to_simulate * self.observation_frequency
            n_samples = int(np.floor(n_samples_dummy))
            last_movement_time = (n_samples_dummy - n_samples) / self.observation_frequency
            movement_time = 1 / self.observation_frequency
                
            for i in range(n_samples):
                self.motion_model(movement_time)
                self.observation_model()
                #print(i)
            
            self.motion_model(last_movement_time)
            if last_movement_time > 0:
                #print(n_samples_dummy, n_samples, 1/self.observation_frequency, last_movement_time)
                self.res_movement_time = 1/self.observation_frequency - last_movement_time
            #print(time_to_simulate, " | ", n_samples_dummy, " | ", movement_time, " | ",\
            #      last_movement_time, " | ", self.res_movement_time, " : ", old_res_movement_time)
        #if self.is_reached():

    def plan_next_waypoint(self, t):
        tic()
        if self.decision_making_mode == "random":
            self.waypoint_end = np.random.rand(1,2)[0] * (self.arena_ub-self.arena_lb) + self.arena_lb
        elif self.decision_making_mode == "corr-random-walk":
            # Based on Experimental comparison of random search strategies for 
            # multi-robot based odour finding without wind information
            rho = 0.2
            mu = 2
            step_length_max = 0.2
            r = np.random.rand()
            dtheta = 2 * np.arctan2( (1-rho)*np.tan(np.pi * (r-0.5)), (1+rho) ) 
            if np.size(self.waypoint_end) > 1:
                theta = dtheta + np.arctan2(self.waypoint_end[1]-self.waypoint_start[1], self.waypoint_end[0]-self.waypoint_start[0])
            else:
                theta = dtheta

            step_length = np.random.rand() * step_length_max
            displacement = np.array([step_length * np.cos(theta), step_length * np.sin(theta)])
            print(displacement, self.waypoint_end)
            dummy_waypoint_end = self.location + displacement
            self.waypoint_end = np.clip(dummy_waypoint_end, self.arena_lb, self.arena_ub)
        elif self.decision_making_mode == "levy-walk":
            # Based on Experimental comparison of random search strategies for 
            # multi-robot based odour finding without wind information
            rho = 0.5
            mu = 2
            step_length_max = 0.2
            theta = np.random.rand() * 2 * np.pi

            r = np.random.rand()
            step_length = step_length_max * np.power(r, 1/(1-mu))
            displacement = np.array([step_length * np.cos(theta), step_length * np.sin(theta)])
            print(displacement, self.waypoint_end)
            dummy_waypoint_end = self.location + displacement
            self.waypoint_end = np.clip(dummy_waypoint_end, self.arena_lb, self.arena_ub)
        elif self.decision_making_mode == "levy-walk-crw":
            # Based on Experimental comparison of random search strategies for 
            # multi-robot based odour finding without wind information
            rho = 0.2
            mu = 2
            step_length_max = 0.2
            r = np.random.rand()
            dtheta = 2 * np.arctan2( (1-rho)*np.tan(np.pi * (r-0.5)), (1+rho) ) 
            if np.size(self.waypoint_end) > 1:
                theta = dtheta + np.arctan2(self.waypoint_end[1]-self.waypoint_start[1], self.waypoint_end[0]-self.waypoint_start[0])
            else:
                theta = dtheta

            r = np.random.rand()
            step_length = step_length_max * np.power(r, 1/(1-mu))
            displacement = np.array([step_length * np.cos(theta), step_length * np.sin(theta)])
            print(displacement, self.waypoint_end)
            dummy_waypoint_end = self.location + displacement
            self.waypoint_end = np.clip(dummy_waypoint_end, self.arena_lb, self.arena_ub)
        else:
            X_robot = []
            y_robot = []
            observation_history = self.get_observations()
            if np.size(observation_history) > 4 and self.decision_counter > 0:
                Xy_unique = np.unique(observation_history[:,:3], axis=0)
                X_robot = Xy_unique[:,:2]
                y_robot = Xy_unique[:,2]
            else:
                X_robot = observation_history[:2]
                X_robot = X_robot.reshape(2,1)
                y_robot = observation_history[2]
            covered_lb, covered_ub = self.get_bounded_space()
            self.decision_making.set_covered_area(covered_lb, covered_ub)
            self.waypoint_end = self.decision_making.get_next_point(t, X_robot, y_robot)
        computing_time = toc()

        if self.penalty_analysis_enable and t > 45:
            if self.id == 0:
                scale = 10
                if self.decision_counter == 0:
                    self.waypoint_end = [1.*scale, .5*scale]
                    if self.id == 0:
                        self.waypoint_end = self.waypoint_end #[0.9, 1]
                    else:
                        scale_2 = 1.4 * scale
                        self.waypoint_end = [1.*scale_2, .5*scale_2]
                else:
                    self.perform_penalty_analysis()#self.waypoint_end)
                    quit()
                    #if self.id == 1:
                    #    quit()

        self.waypoint_start = self.location
        self.waypoint_history = np.vstack((self.waypoint_history, self.waypoint_end))
        
        self.decision_counter += 1
        self.update_movement_direction()
        self.log_decision_making(computing_time)
        
        dx = self.waypoint_end - self.waypoint_start
        self.robot_heading = dx/np.linalg.norm(dx)

        # @note: It can be used to seed observations into the case
        # if self.penalty_analysis_enable:
        #     X = [[.5*scale, 1.5*scale], [0.4*scale, 1.9*scale]]
        #     for x in X:
        #         sensor_value = self.source.measure(x)
        #         dumy_observation_last = np.hstack((x, sensor_value, self.id))
        #         self.observation_history = np.vstack((self.observation_history, dumy_observation_last))
        #print(self.decision_counter)
        
        return self.waypoint_end
        
    def get_robot_plan(self):

        return np.hstack((self.waypoint_start, self.waypoint_end))
    
    def share_information(self):
        plan = self.get_robot_plan()
        belief_model = self.decision_making.get_expected_source_info()
        known_fake_source = self.decision_making.get_local_source()
        data_packet = {"plan": plan, "observations": self.observation_last_waypoint, "belief_model": belief_model, "known_fake_source": known_fake_source}
        return data_packet 

    def receive_information(self, t, data_packet):
        robot_name = data_packet["robot_name"]
        data_observation = data_packet["observations"]
        plan = data_packet["plan"]
        belief_model = data_packet["belief_model"]
        known_fake_source = data_packet["known_fake_source"]
        self.data_packets_all[robot_name]["timestamp"] = data_packet["timestamp"]
        self.data_packets_all[robot_name]["observations"] = data_observation
        self.data_packets_all[robot_name]["plan"] = plan
        self.data_packets_all[robot_name]["belief_model"] = belief_model
        self.data_packets_all[robot_name]["known_fake_source"] = known_fake_source
        self.peers_plan[robot_name] = plan
        if self.is_filtering_over_received_information:
            # x_star = belief_model["xbest"]
            # p = (belief_model["mu"], belief_model["sigma"])
            # mu2, sig2 = self.decision_making.gp_mu.predict(x_star)
            # q = (mu2, sig2)
            # Delta = kl_divergence_norm(x_star, p, q)
            x_star_self = self.decision_making.get_expected_source()
            is_infromative = False
            if np.size(x_star_self) == 0:
                is_infromative = True
            else:
                x_star_peer = belief_model["xbest"]
                dx = (x_star_peer - x_star_self)/self.arena_bound
                Delta = np.linalg.norm(dx)
                if Delta > self.information_gain_threshold:
                    is_infromative = True
            if is_infromative:
                self.observation_history = np.vstack((self.observation_history, data_observation))
        elif np.size(data_observation) > 0:
            self.observation_history = np.vstack((self.observation_history, data_observation))
        
        self.decision_making.set_local_source(known_fake_source)

        if np.size(self.observation_history_shared) > 0:
            self.observation_history_shared = np.vstack((self.observation_history_shared, data_observation))
        else:
            self.observation_history_shared = data_observation
        
    def get_bounded_space(self):
        covered_ub = np.max(self.trajectory_history, axis=0)
        covered_lb = np.min(self.trajectory_history, axis=0)
        return covered_lb, covered_ub

    def get_peers_plan(self):

        return self.peers_plan

    def log_decision_making(self,computing_time):
        if np.size(self.decision_computing_time) == 0:
            self.decision_computing_time = computing_time
        else:
            self.decision_computing_time = np.vstack((self.decision_computing_time,computing_time))

    def get_observations(self):
        if self.is_sharing_observation:
            observation_history = self.observation_history
        else:
            observation_history = self.observation_history_self
        
        return observation_history
        
    def get_data(self):
        observation_history = self.get_observations()
        return observation_history, self.decision_counter, self.decision_computing_time

    def motion_model(self, movement_time):
        traveled_distance = self.velocity * movement_time
        #print(self.location, ' V ', self.robot_heading)
        self.location = self.location + self.robot_heading * traveled_distance
        #print(self.location, ' >> ', traveled_distance)
        self.traveled_distance += traveled_distance
        self.trajectory_history = np.vstack((self.trajectory_history, self.location))

    def noise_model(self, noise_level):
        return np.random.normal(0, noise_level)

    def observation_model(self):
        #self.is_enabled_filtering = False
        is_infromative = True
        sensor_value = self.source.measure(self.location)
        if self.measurement_noise_rate > 0:
            sensor_value += self.noise_model(self.measurement_noise_rate * sensor_value)
        self.observation_last = np.hstack((self.location, sensor_value, self.id))
        
        if self.is_enabled_filtering:
            x_star = self.observation_last[:2]
            y_star = self.observation_last[2]
            gp_model = self.decision_making.gp_mu
            gp_model_extended = self.decision_making.gp_mu_extended
            if np.size(self.observation_history) > 4:
                is_infromative = self.filtering.filter_infomration(gp_model, gp_model_extended, x_star, y_star)
            
        if is_infromative:
            if np.size(self.observation_last_waypoint) == 0:
                self.observation_last_waypoint = self.observation_last
            else:
                self.observation_last_waypoint = np.vstack((self.observation_last_waypoint, self.observation_last))
            self.observation_history = np.vstack((self.observation_history, self.observation_last))
            self.observation_history_self = np.vstack((self.observation_history_self, self.observation_last))
        if self.is_enabled_full_log:
            self.observation_history_full = np.vstack((self.observation_history, self.observation_last))
        
    def update_movement_direction(self):
        delta_waypoints = self.waypoint_end-self.waypoint_start
        distance_waypoints = np.linalg.norm(delta_waypoints)
        if distance_waypoints > self.movement_resolution:
            self.movement_direction = delta_waypoints/distance_waypoints
        else:
            self.movement_direction = np.zeros((1,self.search_space_dimension))

        return self.movement_direction
    
    def get_time_to_reach(self):
        dist = np.linalg.norm(self.waypoint_end - self.location)
        if self.velocity > 0:
            time_to_reach = dist / self.velocity
        else:
            time_to_reach = float('nan')

        return time_to_reach

    def get_time_to_source(self):
        is_on_the_path = False
        self.movement_line = get_line_equation(self.waypoint_start, self.waypoint_end)
        dist_separation = get_distance_from_line(self.movement_line, self.source_location)
        time_to_source = float('nan')
        if dist_separation <= self.source_detection_range: # Robot found the source.
            is_on_the_path, dist_to_source = check_point_is_between_two_points(self.waypoint_start, self.waypoint_end, self.source_location)
            if is_on_the_path and self.velocity > 0:
                time_to_source = dist_to_source / self.velocity
            
        return is_on_the_path, time_to_source

    def check_found_source(self):
        is_found_source = False
        dist_to_source = np.linalg.norm(self.source_location - self.location)
        if dist_to_source <= self.source_detection_range: # Robot found the source.
            print("Robot ", self.id, " found the source!\n")
            is_found_source = True
        self.is_found_source = is_found_source
        self.dist_to_source = dist_to_source

        return [is_found_source, dist_to_source]

    def is_reached_time(self):
        is_reached_to_waypoint = False
        if self.get_time_to_reach() <= self.reached_time_resolution:
            is_reached_to_waypoint = True
        
        return is_reached_to_waypoint

    def is_reached(self):
        is_reached_to_waypoint = False
        dist_to_waypoint = np.linalg.norm(self.waypoint_end - self.location)
        if dist_to_waypoint <= 1e-3: # Robot found the target waypoint.
            is_reached_to_waypoint = True
        
        return is_reached_to_waypoint

    def update_reached_waypoint(self):
        self.observation_last_waypoint = []
        
    def get_sensor_info(self):
        
        return self.measurement_noise_rate, self.observation_frequency 
        
    def get_robot_velocity(self):
        
        return self.velocity

    def get_robot_position(self):
        
        return self.location, self.robot_heading
    
    def get_robot_id(self):
        
        return self.id

    def get_n_robots(self):
        
        return self.n_robots
    
    def get_robot_trajectory(self):

        return self.trajectory_history[:,:2]

    def get_robot_waypoints(self):

        return self.waypoint_history

    def get_belief_model(self):
        
        return self.decision_making.gp_mu

    def get_knowledge_uncertainty_model(self):
        
        return self.decision_making.gp_sigma

    def get_expected_source(self):

        return self.decision_making.get_expected_source()

    def gp_contour_data(self, model, X1, X2):
        N, _ = np.shape(X1)
        X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
        if model == "belief":
            Y_mean, Y_std = self.decision_making.gp_mu.predict(X)
        else:
            Y_mean, Y_std = self.decision_making.gp_sigma.predict(X)

        Y_mean = Y_mean.reshape(N,-1)
        Y_std = Y_std.reshape(N,-1)

        return Y_mean, Y_std

    def get_ideal_time(self):
        if self.velocity == 0:
            ideal_time = float('nan')
        else:
            ideal_time = np.linalg.norm(self.trajectory_history[0,:] - self.source_location) / self.velocity

        return ideal_time

    def save_time_profiling(self, file_name):
        self.decision_making.save_time_profiling(file_name)

    def perform_penalty_analysis(self, next_location=[]):
        X1, X2, Y = self.source.get_data_for_plot()
        X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
        n_data_points = len(X)
        Y_acquisition = np.zeros((n_data_points,))
        Y_acquisition_base = np.zeros((n_data_points,))
        Y_acquisition_actual = np.zeros((n_data_points,))

        for i in range(n_data_points):
            ## Calculate Bayes-Swarm-II's Acquisition Function (with Penalty Term)
            Y_acquisition[i] = self.decision_making.acquisition_function(X[i,:])
            ## Calculate Bayes-Swarm's Acquisition Function (witout Penalty Term)
            Y_acquisition_base[i] = self.decision_making.acquisition_function(X[i,:], bayes_swarm_mode="base")
            if self.id == 0:
                Y_acquisition_actual[i] = Y_acquisition_base[i]
        
        L, M = self.decision_making.get_penalty_parameters()
        Y_mean, Y_std = self.gp_contour_data("belief", X1, X2)
        x_peer = []
        X_peers = []
        ## Calculate Actual Batch Acquisition Function that has been estimated by Bayes-Swarm-II
        # Add the first robot's plan as observation and update the belief model
        X_peers_plan = self.get_peers_plan()
        if self.id == 0:
            peer_id = 1
            j_peer = 'robot-'+str(peer_id)
            if len(next_location) > 0:
                X_peers_plan[j_peer][2:] = next_location
            x_peer = X_peers_plan[j_peer][2:]
            y_peer = self.source.measure(x_peer)
            X_robot = []
            y_robot = []
            observation_history = self.get_observations()
            observation_history = np.vstack((observation_history, np.hstack((x_peer, y_peer, self.id))))
            if np.size(observation_history) > 4 and self.decision_counter > 0:
                Xy_unique = np.unique(observation_history[:,:3], axis=0)
                X_robot = Xy_unique[:,:2]
                y_robot = Xy_unique[:,2]
            else:
                X_robot = observation_history[:2]
                X_robot = X_robot.reshape(2,1)
                y_robot = observation_history[2]
            
            if len(X_peers_plan) > 0:
                i = 0
                for i_peer in X_peers_plan:
                    x_peer_plan = X_peers_plan[i_peer]
                    if np.size(x_peer_plan) > 0:
                        if i == 0:
                            X_peers_dummy = self.decision_making.get_interpoints(x_peer_plan[:2], x_peer_plan[2:])
                            if np.size(X_peers_dummy) > 0:
                                X_peers = X_peers_dummy
                                i += 1
                        else:
                            X_peers_dummy = self.decision_making.get_interpoints(x_peer_plan[:2], x_peer_plan[2:])
                            if np.size(X_peers_dummy) > 0:
                                X_peers = np.vstack((X_peers, X_peers_dummy))    
        
            self.decision_making.update_model(X_robot, y_robot, X_peers)
            for i in range(n_data_points):
                Y_acquisition_actual[i] = self.decision_making.acquisition_function(X[i,:], bayes_swarm_mode="base")
            
        Y_mean_updated, Y_std_updated = self.gp_contour_data("belief", X1, X2)

        penalty_analysis_data = {"Robot_id": self.id, "robot_loc": self.location, "peer_loc": x_peer, "X1": X1, "X2": X2, "Y": Y, "Y_mean": Y_mean, "Y_std": Y_std, 
                                "Y_mean_updated": Y_mean_updated, "Y_std_updated": Y_std_updated,
                                "Y_acquisition": Y_acquisition, "Y_acquisition_base": Y_acquisition_base,
                                "Y_acquisition_actual": Y_acquisition_actual, "L": L, "M": M}

        file_name = "postprocessing/data/results_penalty_analysis_"+str(self.id)
        with open(file_name+'.pickle', 'wb') as handle:
            pickle.dump(penalty_analysis_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
