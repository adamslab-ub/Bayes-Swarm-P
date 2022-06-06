#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/25/2020 """

import numpy as np 
from time import time
from scipy.stats import norm

from BayesSwarm.util import kl_divergence_norm


class Filtering:
    def __init__(self, prediction_threshold=0.5, prediction_score=True, information_score=False):
        self.is_enabled_prediction_score = prediction_score
        self.is_enabled_information_gain_score = information_score
        self.prediction_threshold = prediction_threshold
        self.Delta = -1
        self.delta = -1
        self.jitter = 1e-10

    def filter_infomration(self, gp_model, gp_model_extended, x_star, y_star):
        is_informative = False
        self.gp_model = gp_model
        self.gp_model_extended = gp_model_extended
        if self.is_enabled_prediction_score:
            is_informative = self.prediction_score(x_star, y_star)
        if not is_informative and self.is_enabled_information_gain_score:
            is_informative = self.information_gain_score(x_star, y_star)
        
        return is_informative

    def prediction_score(self, x_star, y_star):
        is_informative = False
        delta0 = self.prediction_threshold
        if self.gp_model.get_training_status():
            mu_star, sigma_star = self.gp_model.predict(x_star)
            #self.delta = np.abs(y_star-mu_star)
            y_star = self.jitter if y_star == 0 else y_star
            self.delta = np.abs((y_star-mu_star)/y_star) 
            #print(delta, 2*delta0*sigma_star**2) 
            #if np.max(self.delta - 2*delta0*sigma_star, 0) > 0:
            if np.max(self.delta - delta0, 0) > 0:
                is_informative = True
        else:
            is_informative = True
        
        return is_informative
    
    def information_gain_score(self, x_star, y_star):
        is_informative = False

        #k = 2 #dim
        if self.gp_model.get_training_status():
            mu1, sig1 = self.gp_model(x_star)
            self.gp_model_extended.update(x_star)
            mu2, sig2 = self.gp_model_extended(x_star)

            q = (mu1, sig1)
            p = (mu2, sig2)
            #det_1 = np.linalg.det(Sigma_1)
            #det_2 = np.linalg.det(Sigma_2)
            #Sigma_2_inv = np.linalg.inv(Sigma_2)
            #trace_Simgas = np.trace(Sigma_2_inv * Sigma_1)
            #delta_mu = mu_2 - mu_1
            #mu_sig_mu = delta_mu.transpose() * Sigma_2_inv * delta_mu
            #Delta = 0.5 * (np.log(det_2/det_1) - k + trace_Simgas + mu_sig_mu)
            self.Delta = kl_divergence_norm(x_star, p, q)
        
            if self.Delta > 0.1:
                is_informative = True
        
        return is_informative

    def get_scores(self):
        score = {"predicition": self.delta, "information_gain": self.Delta}
        return score

def downsample(X, y, active_set_size=1000):
    """Decrease sample rate by integer factor.
    
    Args:
        X (matrix, float): The location of samples
        y (vector, float): The observation values of samples
    
    Returns:
        matrix, vector: The location and observation values of the down-sampled data set
    """
    sample_size = np.size(y)
    if sample_size > active_set_size:
        N = int(np.floor(sample_size/active_set_size))
        idx = np.arange(0, sample_size-1, N)# 0, -N)
        Xe = X[idx,:]
        ye = y[idx]
    else:
        Xe = X
        ye = y
    
    return Xe, ye
