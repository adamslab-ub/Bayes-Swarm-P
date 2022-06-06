#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/24/2020 """

import numpy as np

from mxfusion import Model, Variable
from mxfusion.components.variables import PositiveTransformation
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.modules.gp_modules import SVGPRegression

import mxnet as mx
from mxfusion.inference import GradBasedInference, MAP, MinibatchInferenceLoop


class SvgpModeling:
    def __init__(self, measurement_noise, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, area_largest_length=50, data_size_max=1000):
        self.is_trained = False
        self.data_size_max = data_size_max
        length_scale_bounds=(1e-05, area_largest_length*10)

        if measurement_noise > 0:
            alpha = measurement_noise
        else:
            alpha = 1e-10
        
        M = 50  # Number of inducing locations

        # Default 
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 1))
        m.noise_var = Variable(shape=(1,), transformation=PositiveTransformation(), initial_value=0.01)
        m.kernel = RBF(input_dim=1, variance=1, lengthscale=1)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=m.kernel, noise_var=m.noise_var, shape=(m.N, 1), num_inducing=M)
        m.Y.factor.svgp_log_pdf.jitter = 1e-6
        self.gp = m
        
    def predict(self, x):
        if np.size(np.shape(x)) == 1:
            x = [x]
        
        m = self.gp
        from mxfusion.inference import TransferInference, ModulePredictionAlgorithm
        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y]),
                                    infr_params=infr.params)
        m.Y.factor.svgp_predict.jitter = 1e-6
        
        return y_mean, y_std

    def update(self, X, y):
        #print("GP: ", np.shape(X), np.shape(y))
        n, _ = np.shape(X)
        n_max = self.data_size_max
        m = self.gp
        m.X = X
        m.Y = y
        infr = GradBasedInference(inference_algorithm=MAP(model=m, observed=[m.X, m.Y]),
                          grad_loop=MinibatchInferenceLoop(batch_size=10, rv_scaling={m.Y: 1000/10}))
        infr.initialize(X=(1000,1), Y=(1000,1))
        infr.params[m.Y.factor.inducing_inputs] = mx.nd.array(np.random.randn(50, 1), dtype='float64')
        infr.run(X=mx.nd.array(X, dtype='float64'), Y=mx.nd.array(Y, dtype='float64'),
                max_iter=50, learning_rate=0.1, verbose=True)
        
                
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
