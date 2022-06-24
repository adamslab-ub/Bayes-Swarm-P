#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import os
import sys 

import numpy as np
import pickle
from matplotlib import pyplot as plt

from scipy.io import savemat

from datetime import date

from BayesSwarm.robot import Robot
from BayesSwarm.source import Source
from BayesSwarm.network import Network

import pybullet as p
import pybullet_data
import time
import yaml

class Simulator:
    def __init__(self, n_robots, source_id, start_locations=None, decision_making_mode="bayes-swarm",
            bayes_swarm_mode='scalable', alpha_mode='adaptive_time', filtering_mode="full", decision_horizon=None,
            velocity=None, observation_frequency=1, optimizers=[None, None], enable_full_observation=True,
            is_scout_team=False, debug=False, time_profiling_enable=False, measurement_noise_rate=0,
            depot_mode="single-depot", enable_log_simulation=True, simulation_configs=None):
        self.simulation_mode = simulation_configs.mode
        self.environment = simulation_configs.environment
        self.texture = simulation_configs.texture
        self.robot_type = simulation_configs.robot_type
        self.time_profiling_enable = time_profiling_enable
        self.debug = debug
        self.enable_log_simulation = enable_log_simulation
        self.is_save_fig = True
        self.is_plot_per_decision = True
        self.reached_robot_id = None
        self.is_scout_team = is_scout_team
        self.n_robots = n_robots
        self.source_id = source_id
        self.enable_plot = True
        self.source = Source(source_id)

        self.angular_range, self.arena_lb, self.arena_ub = self.source.get_source_info_arena()
        self.source_location, self.time_max = self.source.get_source_info_mission()
        self.velocity, self.decision_horizon,\
            self.decision_horizon_init, self.source_detection_range = self.source.get_source_info_robot()
        if np.size(start_locations) > 1:
            if start_locations.all == None:
                self.start_locations = self.compute_start_locations(depot_mode)
            else:
                self.start_locations = start_locations
        elif start_locations == None:
            self.start_locations = self.compute_start_locations(depot_mode)


        self.local_penalizing_coef = self.source.get_source_bayes_settings()
        communication_range = self.source.get_source_communication_range()
        if velocity != None:
            self.velocity = velocity
            self.source.set_velocity(velocity)

        if decision_horizon != None:
            self.decision_horizon = decision_horizon
            self.source.set_decision_horizon(decision_horizon)
        
        if observation_frequency > 0:
            self.observation_frequency = observation_frequency # [Hz]
        else:
            raise('Observation frequency must be a positive value!')

        self.alpha_mode = alpha_mode
        
        self.robots = {}
        self.time_to_source = {}
        self.time_to_reach = {}
        self.found_source = {}
        self.is_on_the_path = {}
        self.decision_making_mode = decision_making_mode
        local_penalizing_coef = self.source.get_source_bayes_settings()
        self.bayes_swarm_args = {"bayes_swarm_mode": bayes_swarm_mode, "local_penalizing_coef": local_penalizing_coef,\
                                "decision_making_mode": decision_making_mode, "filtering_mode": filtering_mode,\
                                "enable_full_observation": enable_full_observation, "time_profiling_enable": time_profiling_enable,
                                "optimizers": optimizers, "depot_mode": depot_mode}
        self.is_full_observation = enable_full_observation
        self.model_final = {}

        self.network = Network(n_robots,
                               is_full_observation=enable_full_observation,
                               communication_range=communication_range)

        # Initiate PyBullet
        self.simultion_motion_mode = "teleport"  # Options: "teleport" "move"
        if self.simulation_mode == "pybullet":
            physicsClient = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
            p.setGravity(0,0,-10)
            shift = [0, -0.02, 0]


            self.robot_body = {}

            config_file = ""
        
            if self.environment == "plain":
                config_file = "BayesSwarm/configs/plain.yaml"
                planeId = p.loadURDF("plane.urdf")
            elif self.environment == "plain-texture":
                config_file = "BayesSwarm/configs/plain.yaml"
                planeId = p.loadURDF("plane.urdf")
            elif self.environment == "building":
                config_file = "BayesSwarm/configs/building.yaml"
            elif self.environment ==  "buiding-texture":
                config_file = "BayesSwarm/configs/building.yaml"
            elif self.environment == "mountain-1":
                config_file = "BayesSwarm/configs/mountain1.yaml"

            # environment config file
            config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
            print(config)

            # Set camera position and orientation
            p.resetDebugVisualizerCamera(cameraDistance=config["camera_distance"], 
                                cameraYaw=config["camera_yaw"], 
                                cameraPitch=config["camera_pitch"], 
                                cameraTargetPosition=config["camera_target_position"])

            meshScale = [1, 1, 1]
            for i in range(len(meshScale)):
                meshScale[i] *= config["scale_factor"]

            env_file = config["env_file"]
            robot_file = ""
            if self.robot_type == "uav":
                robot_file = "BayesSwarm/object_files/drone.obj"
                self.elevation = config["elevation_uav"]
            elif self.robot_type == "ugv":
                robot_file = "BayesSwarm/object_files/Cylinder.obj"
                self.elevation = config["elevation_ugv"]

            self.visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=env_file,
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)

            robotMeshScale = [0.02, 0.02, 0.02]
            self.visualShapeId1 = p.createVisualShape(shapeType=p.GEOM_MESH,
                    fileName=robot_file,
                    rgbaColor=[1, 0, 0, 1],
                    specularColor=[0.4, .4, 0],
                    visualFramePosition=shift,
                    meshScale=robotMeshScale)
            
            # Initiate env
            rotation_vector = [0, 0, 0] 
            for i in range(len(rotation_vector)):
                rotation_vector[i] = config["rotation_vector_env"][i]*np.pi/180
            self.cubeStartOrientation = p.getQuaternionFromEuler(rotation_vector)

            self.env = p.createMultiBody(baseMass=0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=self.visualShapeId,
                basePosition=config["origin_env"],
                baseOrientation=self.cubeStartOrientation,
                useMaximalCoordinates=True)
        
            if self.texture == "source":
                textureId = p.loadTexture("BayesSwarm/object_files/beacon.png")
                p.changeVisualShape(objectUniqueId=self.env, linkIndex=-1, textureUniqueId=textureId)

        # Initialize each robot
        if enable_full_observation:
            self.robots_location = None
        else:
            self.robots_location = {}
        self.robots_list = []
        self.mission_metrics = {}
        self.mission_metrics_final = {"mission_time": float("nan"),
                                      "mapping_error": float("nan"),
                                      "computing_time_med": float("nan"),
                                      "computing_time_max": float("nan")}

        for robot_id in range(n_robots):
            robot_name = 'robot-'+str(robot_id)
            self.robots_list = np.append(self.robots_list, robot_name)
            velocity = self.velocity
            start_location = self.start_locations[robot_id,:]
            source_detection_range = self.source_detection_range
            observation_frequency = self.observation_frequency
            signal_source = self.source
            if self.is_scout_team and robot_id == 0:
                tmp_bayes_swarm_args = {"bayes_swarm_mode": "explorative-penalized",
                                        "local_penalizing_coef": local_penalizing_coef,
                                        "decision_making_mode": decision_making_mode,
                                        "filtering_mode": filtering_mode, "optimizers": optimizers,
                                        "enable_full_observation": enable_full_observation,
                                        "time_profiling_enable": time_profiling_enable}
                self.robots[robot_name] = Robot(robot_id, n_robots, tmp_bayes_swarm_args, signal_source,
                                                velocity, start_location, source_detection_range,
                                                observation_frequency, measurement_noise_rate)
            else:
                self.robots[robot_name] = Robot(robot_id, n_robots, self.bayes_swarm_args, signal_source,
                                                velocity, start_location, source_detection_range,
                                                observation_frequency, measurement_noise_rate)
            self.time_to_source[robot_name] = float('nan')
            self.time_to_reach[robot_name] = float('nan')
            self.is_on_the_path[robot_name] = False
            self.found_source[robot_name] = [False, float('nan')]
            self.mission_metrics[robot_name] = [-1, -1, -1, -1]  # Time, MAE, Computing Time, Counter

            # Initiate robots
            if self.simulation_mode == "pybullet":
                rotation_vector = [0, 0, 0] 
                for i in range(len(rotation_vector)):
                    rotation_vector[i] = config["rotation_vector_robot"][i]*np.pi/180
                self.cubeStartOrientation = p.getQuaternionFromEuler(rotation_vector)
                
                robot_position = [start_location[0], start_location[1], self.elevation]  
                if self.simultion_motion_mode == "teleport":
                    self.robot_body[robot_name] = p.createMultiBody(baseMass=1,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseVisualShapeIndex=self.visualShapeId1,
                                        basePosition=robot_position,
                                        baseOrientation=self.cubeStartOrientation,
                                        useMaximalCoordinates=True)
                else:
                    self.object_id = p.loadURDF("BayesSwarm/object_files/arial_vehicle.urdf", basePosition=robot_position, baseOrientation=self.cubeStartOrientation)
                    self.robot_body[robot_name] = p.createConstraint(self.object_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],[0, 0, 0], [0,0,0])
                    p.changeConstraint(self.robot_body[robot_name], robot_position)

                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                p.setGravity(0, 0, 0)
                p.setRealTimeSimulation(1)
                p.stepSimulation()
                time.sleep(1)
        dir_name = 'output'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        today = date.today().strftime("%Y%b%d")  
        self.file_name = dir_name + '\\results_' + today + '_' + str(np.random.randint(0,1000))
        if self.is_save_fig:
            self.fig_counter = 0

    def run(self):
        t = 0
        is_successed = False
        robot_successed = 0

        ## First decision-making 
        # robots_list = self.robots:
        robots_list = list(self.robots_list) #np.random.shuffle(self.robots_list) #flip

        if not self.is_full_observation:
            for irobot in robots_list:
                self.robots_location[irobot], _ = self.robots[irobot].get_robot_position()
                
        for irobot in robots_list: #self.robots:
            self.robots[irobot].plan_next_waypoint(t)
            if self.decision_making_mode == "bayes-swarm": 
                data_packet = self.robots[irobot].share_information()
                self.network.broadcast_information(t, irobot, data_packet)
                neighbours_list = self.network.get_neighbours_list(irobot, self.robots_location)
                for iirobot in neighbours_list:
                    data_packet = self.network.get_information()
                    self.robots[iirobot].receive_information(t, data_packet)

            self.robots[irobot].update_reached_waypoint()
            observation_history, decision_counter, _ = self.robots[irobot].get_data()
            print("+ with Observations: ", np.shape(observation_history), " and ", decision_counter," decisions ++++++++++++++++")
        

        ## Next decision-making
        is_not_terminated = True
        #self.time_max = 40
        while is_not_terminated:
            if t >= self.time_max:
                is_not_terminated = False
                break

            ## Moving robots based on the smallest allowed travel time (earlieast time one robot reaches to its waypoint).
            time_to_simulate, robot_key = self.get_smallest_allowed_travel_time()
            if np.isnan(time_to_simulate):
                raise("Invalid time to simulate")

            print("{} needs {:#.3f} secs".format(robot_key, time_to_simulate))
            for irobot in robots_list:
                print("+ {} @ t = {:#.3f} secs ++++++++++++++++".format(irobot, t))
                #robot_location, _ = self.robots[irobot].get_robot_position()
                self.robots[irobot].step(time_to_simulate)
                observation_history, decision_counter, _ = self.robots[irobot].get_data()
                print("+ with Observations: {} and {} decisions ++++++++++++++++".format(np.shape(observation_history), decision_counter))
                #robot_location, _ = self.robots[irobot].get_robot_position()
                #print(irobot, ": ", robot_location,  ":", self.robots[irobot].get_robot_plan())
                #os.system("pause")
                self.found_source[irobot] = self.robots[irobot].check_found_source()
                if self.found_source[irobot][0] == True:
                    is_successed = True
                    is_not_terminated = False
                    robot_successed = irobot
                    robot_dist_to_source = self.found_source[irobot][1]
                    if self.decision_making_mode == "bayes-swarm": 
                        self.model_final["gp_mu"] = self.robots[irobot].get_belief_model()
                        self.model_final["gp_sigma"] = self.robots[irobot].get_knowledge_uncertainty_model()
                    break

                ## Update robot locations in PyBullet
                if self.simulation_mode == "pybullet":
                    location, robot_heading = self.robots[irobot].get_robot_position()
                    robot_position = [location[0], location[1], self.elevation]
                    if self.simultion_motion_mode == "teleport":
                        p.resetBasePositionAndOrientation(bodyUniqueId=self.robot_body[irobot], posObj=robot_position, ornObj=self.cubeStartOrientation)
                    else:
                        robot_position = [location[0], location[1], self.elevation]   
                        p.changeConstraint(self.robot_body[irobot], robot_position)

            t += time_to_simulate
            save_robot_start_loc = False
            if save_robot_start_loc:
                robot_locations = []
                for irobot in robots_list:
                    robot_location, _ = self.robots[irobot].get_robot_position()
                    robot_locations.append(robot_location)
                savemat("BayesSwarm_StartLoc_Case"+str(self.source_id)+".mat", {"start_locations": robot_locations})
                quit()


            ## If robot reached, take decision and share information.
            if is_not_terminated == True:
                if not self.is_full_observation:
                    for irobot in robots_list:
                        self.robots_location[irobot], _ = self.robots[irobot].get_robot_position()
                
                
                for irobot in robots_list:
                    if self.robots[irobot].is_reached() == True:
                        #print(irobot, ", Reached: ", robot_location)
                        #os.system("pause")
                        self.robots[irobot].plan_next_waypoint(t)
                        #os.system("pause")
                        if self.decision_making_mode == "bayes-swarm": 
                            data_packet = self.robots[irobot].share_information()
                            self.network.broadcast_information(t, irobot, data_packet)
                            neighbours_list = self.network.get_neighbours_list(irobot, self.robots_location)
                            for iirobot in neighbours_list:
                                data_packet = self.network.get_information()
                                self.robots[iirobot].receive_information(t, data_packet)
                        if self.debug:
                            self.reached_robot_id = irobot
                            self.plot_robot_trajectory()

            #if t > 45:
            #    self.debug = True

        ## Log print screen in a text file
        stdoutOrigin=sys.stdout 
        sys.stdout = open(self.file_name+"_log.txt", "w")

        print("\n+=======================================================")
        
        # Report Mission Performance
        X1_true, X2_true, Y_true = self.source.get_data_for_plot()
        if is_successed == True:
            ideal_time = self.robots[robot_successed].get_ideal_time()
            if self.decision_making_mode == "bayes-swarm":
                Y_pred, _ = self.robots[robot_successed].gp_contour_data("belief", X1_true, X2_true)
                mapping_error = np.mean(np.abs(Y_true - Y_pred))
            else:
                mapping_error = float("NaN")

            mission_time = np.abs(t-ideal_time)/ideal_time
            _, decision_counter, decision_computing_time = self.robots[robot_successed].get_data()
            self.mission_metrics[robot_successed][:2] = [mission_time, mapping_error] #Time, MAE, Computing Time, COunter
            print("+ {} found the source at time {:#.3f} out of {:#.3f}. It is located {:#.3f} meters of the source.\
                   Ideal time to find this source is {:#.3f}"
                  .format(robot_successed, t, self.time_max, robot_dist_to_source,ideal_time))
            print("+ Mission Performance Metrics -- Completion Time: {:#.3f}; Mapping Error (MAE) {:#.3f}."
                  .format(mission_time, mapping_error))
            self.mission_metrics_final = {"mission_time":mission_time, "mapping_error":mapping_error,
                                          "computing_time_med": np.median(decision_computing_time),
                                          "computing_time_max": np.max(decision_computing_time)}
        else:
            print("+ Timeout ({:#.3f}/{:#.3f}): Search mission failed!".format(t, self.time_max))
            found_source = self.found_source
            key = min(found_source.keys(), key=(lambda k: found_source[k][1]))
            print("+ The closest robot to the target is {} with {:#.3f} meters distance.".format(key, found_source[key][1]))
        print("+-------------------------------------------------------")
        
        for irobot in self.robots:
            _, decision_counter, decision_computing_time = self.robots[irobot].get_data()
            if self.decision_making_mode == "bayes-swarm":
                Y_pred, _ = self.robots[irobot].gp_contour_data("belief", X1_true, X2_true)
                mapping_error = np.mean(np.abs(Y_true - Y_pred))
            else:
                mapping_error = float("NaN")

            self.mission_metrics[irobot][1:] = [mapping_error, np.median(decision_computing_time), decision_counter] #Time, MAE, Computing Time, COunter
            print("+ Robot-{} takes {} decisions with a median computing time {:#.3f} (min: {:#.3f}, max: {:#.3f}); Mapping Error (MAE): {:#.3f}."\
                  .format(irobot, decision_counter, np.median(decision_computing_time),\
                  np.min(decision_computing_time), np.max(decision_computing_time),mapping_error))
        print('+=======================================================\n')


        if self.time_profiling_enable:
            self.robots[robot_successed].save_time_profiling(self.file_name)

        if self.enable_log_simulation:
            self.log_simulation()
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        if self.enable_plot:
            self.plot_robot_trajectory()
        
        print("Simulation ended - ", self.file_name)

    def log_simulation(self):
        simulator_self = self
        with open(self.file_name+'.pickle', 'wb') as handle:
            pickle.dump(simulator_self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def get_smallest_allowed_travel_time(self):
        time_resolution_order = 4
        time_to_source = self.get_all_time_to_source()
        time_to_reach = self.get_all_time_to_reach()
        print("time_to_source: ", time_to_source)
        print("time_to_reach: ", time_to_reach)
        #key_source = min(time_to_source.keys(), key=(lambda k: time_to_source[k]))
        #time_to_source_min = time_to_source[key_source]
        time_to_source = list(time_to_source.values())
        time_to_source = [x for x in time_to_source if str(x) != 'nan']
        if len(time_to_source) > 0:
            time_to_source_min = np.round(min(time_to_source),time_resolution_order)
        else:
            time_to_source_min = float('nan')
        #time_to_source_min = min(val for val in time_to_source.values() if not np.isnan(val))
        #key_reach = min(time_to_reach.keys(), key=(lambda k: time_to_reach[k]))
        #time_to_reach_min = time_to_reach[key_reach]
        key_reach = None
        key_source = None 
        
        time_to_reach = list(time_to_reach.values())
        time_to_reach = [x for x in time_to_reach if str(x) != 'nan']
        if len(time_to_reach) > 0:
            time_to_reach_min = np.round(min(time_to_reach), time_resolution_order)
        else:
            time_to_reach_min = float('nan')

        if np.isnan(time_to_source_min):
            key = key_reach
            time_to_simulate = time_to_reach_min
        else:
            if time_to_reach_min <= time_to_source_min:
                key = key_reach
                time_to_simulate = time_to_reach_min
            else:
                key = key_source
                time_to_simulate = time_to_source_min
        
        return time_to_simulate, key

    def get_all_time_to_source(self):
        for robot in self.robots:
            self.is_on_the_path[robot], self.time_to_source[robot] = self.robots[robot].get_time_to_source()
        return self.time_to_source

    def get_all_time_to_reach(self):
        for robot in self.robots:
            self.time_to_reach[robot] = self.robots[robot].get_time_to_reach()
        
        return self.time_to_reach

    def check_all_found_source(self):
        for robot in self.robots:
            self.found_source[robot] = self.robots[robot].check_found_source()

        return self.found_source

    def get_ideal_times(self):
        dist_distribution = np.linalg.norm(self.start_locations - self.source_location,axis=1)
        time_distribution = dist_distribution / self.velocity

        return np.median(time_distribution), np.mean(time_distribution),\
               np.min(time_distribution), np.max(time_distribution)

    def plot_robot_trajectory(self):
        is_tiled_plots = False
        if is_tiled_plots:
            _, (ax, ax2) = plt.subplots(1, 2)
        else:
            _, ax = plt.subplots()
        X1_true, X2_true, Y_true = self.source.get_data_for_plot()
        color_list = ['red','orange','brown','dodgerblue', 'green','k', 'b', 'r', 'g', 'm', 'k', 'b', 'r', 'g', 'm', 'k', 'b', 'r', 'g', 'm', 'k', 'b', 'r', 'g', 'm', 'k']
        n_color = np.size(color_list)
        stdoutOrigin=sys.stdout 
        sys.stdout = open(self.file_name+"_"+str(self.fig_counter)+".txt", "w")
            
        i = 0
        for irobot in self.robots:
            color = color_list[min(i,n_color-1)]
            i += 1
            robot_instance = self.robots[irobot]
            robots_trajectory = robot_instance.get_robot_trajectory()
            robots_observations = robot_instance.get_observations()
            print("Robot-", irobot, " : ", len(robots_observations))
            ax.plot(robots_trajectory[:,0], robots_trajectory[:,1], '-', color=color, label=irobot)
            waypoints = robot_instance.get_robot_waypoints()
            #ax.plot(waypoints[:,0], waypoints[:,1], '--s', color=color, label=irobot+"-w")
            if self.decision_making_mode == "bayes-swarm":
                expected_source_location = robot_instance.get_expected_source()
                if not expected_source_location == []:
                    ax.plot(expected_source_location[0], expected_source_location[1], 'X', color=color, label=irobot+"-expected-source")
            if self.is_plot_per_decision:
                if self.reached_robot_id == irobot:
                    ax.plot(robots_observations[:,0], robots_observations[:,1], '.', color="k", label=irobot)
                    if self.decision_making_mode == "bayes-swarm":
                        Y_pred, _ = self.robots[irobot].gp_contour_data("belief", X1_true, X2_true)
                        ax.contour(X1_true, X2_true, Y_pred, linestyles='dashed', colors=color)
                        if is_tiled_plots:
                            _, Y_std = self.robots[irobot].gp_contour_data("uncertainity", X1_true, X2_true)
                            #self.model_final["gp_sigma"] = self.robots[irobot].get_knowledge_uncertainty_model()
                            ax2.contourf(X1_true, X2_true, Y_std)
            else:
                if self.decision_making_mode == "bayes-swarm":
                    _, Y_std = self.robots[irobot].gp_contour_data("uncertainity", X1_true, X2_true)
                    #self.model_final["gp_sigma"] = self.robots[irobot].get_knowledge_uncertainty_model()
                    ax2.contourf(X1_true, X2_true, Y_std)
        ax.contour(X1_true, X2_true, Y_true, colors='gray')
        ax.set_aspect('equal', 'box')
        if is_tiled_plots:
            ax2.set_aspect('equal', 'box')
        
        #ax.legend()
        
        if self.is_save_fig:
            plt.savefig(self.file_name+"_"+str(self.fig_counter)+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
            self.fig_counter += 1

        sys.stdout.close()
        sys.stdout=stdoutOrigin

        #plt.show()

    def get_mission_metrics(self):
        return self.mission_metrics_final

    def compute_start_locations(self, depot_mode="single-depot"):
        n_robots = self.n_robots
        arena_lb = self.arena_lb
        arena_ub = self.arena_ub
        if depot_mode == "four-depot":
            start_locations = np.zeros((n_robots,2))
            n_robot_per_corner = n_robots // 4
            # Lower left corner
            location = np.array([self.arena_lb[0], self.arena_lb[1]])
            l_index = 0
            u_index = n_robot_per_corner
            start_locations[:u_index, :] = np.tile(location, (n_robot_per_corner, 1))
            # Lower right corner
            location = np.array([self.arena_ub[0], self.arena_lb[1]])
            l_index = n_robot_per_corner
            u_index = 2 * n_robot_per_corner
            start_locations[l_index:u_index, :] = np.tile(location, (n_robot_per_corner, 1))
            # Upper right corner
            location = np.array([self.arena_ub[0], self.arena_ub[1]])
            l_index = 2 * n_robot_per_corner
            u_index = 3 * n_robot_per_corner
            start_locations[l_index:u_index, :] = np.tile(location, (n_robot_per_corner, 1))
            # Upper left corner
            location = np.array([self.arena_lb[0], self.arena_ub[1]])
            l_index = 3 * n_robot_per_corner
            n_robot_per_corner = n_robots - 3 * n_robot_per_corner
            start_locations[l_index:, :] = np.tile(location, (n_robot_per_corner, 1))
        else:
            start_locations = np.zeros((n_robots,2))
        return start_locations
