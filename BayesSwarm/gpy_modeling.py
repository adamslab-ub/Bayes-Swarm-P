#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/24/2020 """

import numpy as np

import GPy


class GpyModeling:
    def __init__(self, measurement_noise, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, area_largest_length=50, data_size_max=1000):
        self.is_trained = False
        self.data_size_max = data_size_max
        length_scale_bounds=(1e-05, area_largest_length*10)

        if measurement_noise > 0:
            alpha = measurement_noise
        else:
            alpha = 1e-8
        
        self.measurement_noise = measurement_noise
        self.M = 20  # Number of inducing locations
        
        # Default 
        
    def predict(self, x):
        if np.size(np.shape(x)) == 1:
            x = np.array([x])
          
        y_mean, y_var = self.gp.predict(x)
        y_std = np.sqrt(y_var)
        y_mean = y_mean.reshape(1,-1)
        y_std = y_std.reshape(1,-1)
        return y_mean[0], y_std[0]

    def update(self, X, y):
        #print("GP: ", np.shape(X), np.shape(y))
        n, _ = np.shape(X)
        n_max = self.data_size_max

        Z = np.random.rand(self.M,1)
        y = y.reshape(-1,1)
        Z = np.array(Z)
        gp = GPy.core.SVGP(X, y, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(), batchsize=10)
        gp.kern.white.variance = self.measurement_noise ** 2
        
        #gp.optimizeWithFreezingZ()
        gp.optimize('bfgs', max_iters=100)

        self.gp = gp
        self.is_trained = True

    def get_gp_model(self):

        return self.gp

    def set_hyperparameters(self, hyperparameters):
        pass

    def get_hyperparameters(self, hyperparameters):
        return self.gp.get_params()

    def get_training_status(self):
        return self.is_trained
