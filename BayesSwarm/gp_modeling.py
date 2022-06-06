#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/22/2020 """

import numpy as np

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from BayesSwarm.filtering import downsample


class GpModeling:
    def __init__(self, measurement_noise, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, area_largest_length=50, active_set_size=1000):
        self.is_trained = False
        self.active_set_size = active_set_size
        length_scale_bounds=(1e-05, area_largest_length*10)

        if measurement_noise > 0:
            alpha = measurement_noise ** 2
        else:
            alpha = 1e-8

        kernel = ConstantKernel() * RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds)

        # Default 
        self.gp = gaussian_process.GaussianProcessRegressor(alpha=alpha, copy_X_train=True,\
                    kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, optimizer='fmin_l_bfgs_b')
        
        
    def predict(self, x):
        if np.size(np.shape(x)) == 1:
            x = [x]
        y_mean, y_std = self.gp.predict(x, return_std=True)
        return y_mean, y_std

    def update(self, X, y):
        #print("GP: ", np.shape(X), np.shape(y))
        n, _ = np.shape(X)
        n_max = self.active_set_size

        if n >= n_max:
            Xe, ye = downsample(X, y, self.active_set_size)
        else:
            Xe = X
            ye = y
        self.gp.fit(Xe, ye)
        self.is_trained = True
        
    def get_gp_model(self):

        return self.gp

    def set_hyperparameters(self, hyperparameters):
        pass

    def get_hyperparameters(self, hyperparameters):
        return self.gp.get_params()

    def get_training_status(self):
        return self.is_trained
