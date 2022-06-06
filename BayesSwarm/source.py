#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np

class Source:
    def __init__(self, id):
        self.id = id
        self.source_dim = 2
        if self.id == 0:
            self.source_0_init()
        elif self.id == 1:
            self.source_1_init()
        elif self.id == 2:
            self.source_2_init()
        elif self.id == 3:
            self.source_3_init()
        elif self.id == 4:
            self.source_4_init()
        elif self.id == 5:
            self.source_5_init()
        elif self.id == 51:
            self.source_51_init()
        elif self.id == 6:
            self.source_6_init()
        elif self.id == 7:
            self.source_7_init()
        elif self.id == 8:
            self.source_8_init()
        elif self.id == 9:
            self.source_9_init()
        elif self.id == 10:
            self.source_10_init()
        else:
            self.source_1_init()
        
    def get_source_location(self):
        
        return self.source_location

    def measure(self, location):
        if self.id == 0:
            signal_value = self.source_0_measure(location)
        elif self.id == 1:
            signal_value = self.source_1_measure(location)
        elif self.id == 2:
            signal_value = self.source_2_measure(location)
        elif self.id == 3:
            signal_value = self.source_3_measure(location)
        elif self.id == 4:
            signal_value = self.source_4_measure(location)
        elif self.id == 5:
            signal_value = self.source_5_measure(location)
        elif self.id == 51:
            signal_value = self.source_5_measure(location)
        elif self.id == 6:
            signal_value = self.source_6_measure(location)
        elif self.id == 7:
            signal_value = self.source_7_measure(location)
        elif self.id == 8:
            signal_value = self.source_8_measure(location)
        elif self.id == 9:
            signal_value = self.source_9_measure(location)
        elif self.id == 10:
            signal_value = self.source_10_measure(location)
        else:
            signal_value = self.source_1_measure(location)
        
        return signal_value

    def gradient(self, location):
        if self.id == 7:
            gradient_value = self.source_7_gradient(location)
        
        return gradient_value

    def source_0_init(self): # Based on 99 in my Matlab code
        self.source_location = np.array([.5, 0.7])
        self.time_max = 100
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([0,0])
        self.arena_ub = np.array([2.4, 2.4])
        self.source_detection_range = 0.05
        self.velocity = 0.1 # [m/s]
        self.decision_horizon_init = 2
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 50}
        self.communication_range = 2
    
    def source_0_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -3.0
        
        dx1 = x - c
        if np.size(location) > self.source_dim:
            dx11 = np.linalg.norm(dx1, axis=1)**2
        else:
            dx11 = np.dot(dx1,dx1)
        
        f = np.exp(dx11/sig1)

        return f

    def source_1_init(self): # Based on Case 1 in MRS paper, but adopted for IROS2020
        self.source_location = np.array([1.9, 2.3])
        self.time_max = 100
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([0,0])
        self.arena_ub = np.array([2.4, 2.4])
        self.source_detection_range = 0.05
        self.velocity = 0.1 # [m/s]
        self.decision_horizon_init = 10 #10
        self.decision_horizon = 10 #10
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 2
    
    def source_1_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -3.0
        sig2 = -0.5
        
        dx1 = x - c
        dx2 = x - np.array([1.5, .5])
        if np.size(location) > self.source_dim:
            dx11 = np.linalg.norm(dx1, axis=1)**2
            dx22 = np.linalg.norm(dx2, axis=1)**2
        else:
            dx11 = np.dot(dx1,dx1)
            dx22 = np.dot(dx2,dx2)

        f = np.exp(dx11/sig1) + 0.5 * np.exp(dx22/sig2)

        return f

    def source_2_init(self): # Based on Case 2 in MRS paper
        self.source_location = np.array([21, 19])
        self.time_max = 1000
        self.angular_range = np.array([0,2*np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 0.2*5 # [m/s]
        self.decision_horizon_init = 20
        self.decision_horizon = 20
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
    
    def source_2_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130
        sig2 = -40
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, 19], [-15, -15]])
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1)**2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_2x5_init(self): # Swarm Scale Scenario
        self.source_location = np.array([21, 19]) * 5
        self.time_max = 1000
        self.angular_range = np.array([0, 2 * np.pi])
        self.arena_lb = np.array([-120, -120])
        self.arena_ub = np.array([120, 120])
        self.source_detection_range = 0.2
        self.velocity = 0.2 * 10 # [m/s]
        self.decision_horizon_init = 50
        self.decision_horizon = 30
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 60
    
    def source_2x5_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130*5
        sig2 = -40*5
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, 19], [-15, -15]]) * 5
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1) ** 2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_7_init(self): # Based on Case 1 in MRS paper, but adopted for IROS2020
        self.source_location = np.array([1., 2.])
        self.time_max = 100
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([0,0])
        self.arena_ub = np.array([2.4, 2.4])
        self.source_detection_range = 0.05
        self.velocity = 0.1 # [m/s]
        self.decision_horizon_init = 10
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 2} #100
        self.communication_range = 0.5
    
    def source_7_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -3.0
        sig2 = -0.5
        
        dx1 = x - c
        dx2 = x - np.array([2., .5])
        if np.size(location) > self.source_dim:
            dx11 = np.linalg.norm(dx1, axis=1)**2
            dx22 = np.linalg.norm(dx2, axis=1)**2
        else:
            dx11 = np.dot(dx1,dx1)
            dx22 = np.dot(dx2,dx2)

        f = np.exp(dx11/sig1) + 0.5 * np.exp(dx22/sig2)

        return f
    
    def source_7_gradient(self, location):
        c = self.source_location
        x = location
        sig1 = -3.0
        sig2 = -0.5
        ## exp(a(x+b)^2+c) --d/dx--> 2a(x+b)exp(a(x+b)^2+c)
        dx1 = x - c
        dx2 = x - np.array([2., .5])
        if np.size(location) > self.source_dim:
            dx11 = np.linalg.norm(dx1, axis=1)**2
            dx22 = np.linalg.norm(dx2, axis=1)**2
        else:
            dx11 = np.dot(dx1,dx1)
            dx22 = np.dot(dx2,dx2)

        dfx = []
        for i in range(2):
            dfx.append(2*(1/sig1)*(x[i]+dx1[i])*np.exp(dx11/sig1) + 2*(1/sig2)*(x[i]+dx2[i])*0.5*np.exp(dx22/sig2))

        return np.linalg.norm(dfx)

    def source_8_init(self): # Based on Case 2 in MRS paper
        self.source_location = np.array([21, 19])
        self.time_max = 1000
        self.angular_range = np.array([0,2*np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 0.2*5 # [m/s]
        self.decision_horizon_init = 10
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
    
    def source_8_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130
        sig2 = -40
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, 19], [-15, -15]])
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1)**2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_9_init(self): # Based on Case 2 in MRS paper
        self.source_location = np.array([21, 19])
        self.time_max = 1000
        self.angular_range = np.array([0,2*np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 0.2*1 # [m/s]
        self.decision_horizon_init = 100
        self.decision_horizon = 100
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
    
    def source_9_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130
        sig2 = -40
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, 19], [-15, -15]])
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1)**2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_3_init(self):
        '''
        Case Study 1 of krishnanand2006glowworm: 
        Max achievement by 50 robots with Glowworm method: 360iter * 0.01 = 3.6sec
        '''
        scale = 1
        self.dim = 2
        self.scale = scale
        self.source_location = np.array([0., 1.6])*scale
        self.time_max = 10
        self.angular_range = np.array([-np.pi,np.pi])
        self.arena_lb = np.array([-3,-3])*scale
        self.arena_ub = np.array([3,3])*scale
        self.source_detection_range = 0.05*scale
        s = 0.01 # Max movement in each iteration Ref to krishnanand2006glowworm
        step_time = 10/1000 # 10 ms
        self.velocity = scale*s/step_time # [m/s]
        self.decision_horizon_init = 0.1
        self.decision_horizon = 0.1
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
    
    def source_3_measure(self, location):
        #measureModel = @(x)(J1Glowworm2006(x));
        n = np.size(location)
        if n > self.dim:
            x = location[:,0]/self.scale
            y = location[:,1]/self.scale
        else:
            x = location[0]/self.scale
            y = location[1]/self.scale
        f = 3*((1-x)**2)*np.exp(-(x**2)-(y+1)**2)\
            - 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2)\
            - (1/3)*np.exp(-(x+1)**2-y**2)
        
        return f


    def source_4_init(self): # Based on Case 2 in MRS paper
        self.source_location = np.array([21, 19])
        self.time_max = 1000
        self.angular_range = np.array([0,2*np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 0.2 # [m/s]
        self.decision_horizon_init = 20
        self.decision_horizon = 20
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
    
    def source_4_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130
        sig2 = -40
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, 19], [-15, -15]])
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1)**2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_51_init(self): # Beacon Signal
        self.source_location = np.array([0, 0])
        self.time_max = 2400
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([-70, -15])
        self.arena_ub = np.array([70, 1])
        self.source_detection_range = 0.05
        self.velocity = 1 # [m/s]
        self.decision_horizon_init = 60
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 0
        self.log_mode = False
        self.height = 10
    
    def source_5_init(self): # Beacon Signal
        self.source_location = np.array([0, 0])
        self.time_max = 900
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([-20, -20])
        self.arena_ub = np.array([20, 20])
        self.source_detection_range = 0.05
        self.velocity = 1 # [m/s]
        self.decision_horizon_init = 20
        self.decision_horizon = 20
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
        self.log_mode = False
        self.height = 10

    def source_5_measure(self, location):
        c = self.source_location
        x = location
        if np.size(x) > self.source_dim:
            n, m = np.shape(x)
            f = np.zeros_like(x[:,0])
            for i in range(n):
                rx = np.abs(x[i,0])
                ry = np.abs(x[i,1])
                rz = self.height
                A = np.matrix([[2*(rx**2)-(ry**2)-(rz**2), 3*rx*ry, 3*rx*rz],\
                            [3*rx*ry, 2*(ry**2)-(rx**2)-(rz**2), 3*ry*rz],\
                            [3*rx*rz, 3*ry*rz, 2*(rz**2)-(rx**2)-(ry**2)]])
                M = np.matrix([[1],[1],[1]])
                r = np.sqrt((rx**2)+(ry**2)+(rz**2))
                H = np.dot(((1/(4*(np.pi)*(r**5))) * A), M)
                if self.log_mode:
                    f[i] = np.log10(np.linalg.norm(H))
                else:
                    f[i] = np.linalg.norm(H)
        else:    
            rx = np.abs(x[0])
            ry = np.abs(x[1])
            rz = self.height
            A = np.matrix([[2*(rx**2)-(ry**2), 3*rx*ry, rz], [3*rx*ry, 2*(ry**2)-(rx**2), rz], [rz, rz, rz-(rx**2)-(ry**2)]])
            M = np.matrix([[1],[1],[0]])
            r = np.sqrt((rx**2)+(ry**2))
            H = np.dot(((1/(4*(np.pi)*(r**5))) * A), M)
            if self.log_mode:
                f = np.log10(np.linalg.norm(H))
            else:
                f = np.linalg.norm(H)

        return f


    def source_6_init(self): # Beacon Signal
        self.source_location = np.array([0, 0])
        self.time_max = 900
        self.angular_range = np.array([0,np.pi/2])
        self.arena_lb = np.array([-70, -70])
        self.arena_ub = np.array([70, 70])
        self.source_detection_range = 0.05
        self.velocity = 1 # [m/s]
        self.decision_horizon_init = 40
        self.decision_horizon = 40
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20
        self.log_mode = False
    
    def source_6_measure(self, location):
        c = self.source_location
        x = location
        if np.size(x) > self.source_dim:
            n, m = np.shape(x)
            f = np.zeros_like(x[:,0])
            for i in range(n):
                rx = np.abs(x[i,0])
                ry = np.abs(x[i,1])
                rz = 0
                r = np.sqrt(rx**2+ry**2+rz**2)
                y = 110/(np.abs(r)**3+12)
                if self.log_mode:
                    f[i] = np.log10(y)
                else:
                    f[i] = y
        else:    
            rx = np.abs(x[0])
            ry = np.abs(x[1])
            rz = 0
            r = np.sqrt(rx**2+ry**2+rz**2)
            y = 110/(np.abs(r)**3+12)
            if self.log_mode:
                f[i] = np.log10(y)
            else:
                f[i] = y
            
        return f

    def source_10_init(self): # Swarm Scale Scenario
        self.source_location = np.array([-45, -45])
        self.time_max = 1000
        self.angular_range = np.array([0, 2 * np.pi])
        self.arena_lb = np.array([-120, -120])
        self.arena_ub = np.array([120, 120])
        self.source_detection_range = 0.2
        self.velocity = 0.2 * 5 # [m/s]
        self.decision_horizon_init = 100
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 480}
        self.communication_range = 60
    
    def source_10_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -1400
        sig2 = -1000
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [0, 15],\
                          [-19, 10], [21, -5], [15, 15],\
                          [-5, 5], [1, 1], [6, 3]]) * 5
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1) ** 2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f

    def source_11_init(self): # Swarm Scale Scenario
        self.source_location = np.array([55, 45])
        self.time_max = 1000
        self.angular_range = np.array([0, 2 * np.pi])
        self.arena_lb = np.array([-120, -120])
        self.arena_ub = np.array([120, 120])
        self.source_detection_range = 0.2
        self.velocity = 0.2 * 5 # [m/s]
        self.decision_horizon_init = 50
        self.decision_horizon = 30
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 60
    
    def source_11_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -130*50
        sig2 = -40*40
        coef = 0.4
        
        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1)**2
        else:
            dx2 = np.dot(dx,dx)
        f = np.exp(dx2/sig1)
        c_list = np.array([[21,-19], [0, -15], [-5, 20],\
                          [-19, 10], [25, 8], [-15, -15]]) * 5
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i,:]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1)**2
            else:
                dx2 = np.dot(dx,dx)
            f += coef * np.exp(dx2/sig2)
        
        return f
    
    def get_source_info(self):
        
        return self.velocity, self.decision_horizon, self.source_detection_range, self.source_location,\
                self.angular_range, self.time_max, self.arena_lb, self.arena_ub
    
    def get_source_info_arena(self):
        
        return self.angular_range, self.arena_lb, self.arena_ub
    
    def get_source_info_robot(self):
        
        return self.velocity, self.decision_horizon, self.decision_horizon_init, self.source_detection_range

    def get_source_info_mission(self):
        
        return self.source_location, self.time_max

    def get_source_bayes_settings(self):
        
        return self.local_penalizing_coef

    def set_velocity(self, velocity):
        Warning('Default velocity changed from {} to {}'.format(self.velocity, velocity))
        self.velocity = velocity
    
    def set_decision_horizon(self, decision_horizon):
        Warning('Default decision-horizon changed from {} to {}'.format(self.decision_horizon, decision_horizon))
        self.decision_horizon = decision_horizon
    
    def get_source_communication_range(self):
        return self.communication_range

    def get_data_for_plot(self):
        N = 100
        x1 = np.linspace(self.arena_lb[0], self.arena_ub[0], N)
        x2 = np.linspace(self.arena_lb[1], self.arena_ub[1], N)
        X1, X2 = np.meshgrid(x1,x2)
        X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
        Y = self.measure(X)
        Y = Y.reshape(N,-1)

        return X1, X2, Y