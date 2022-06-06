#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np 

from time import time

from scipy.stats import norm

tics = []
def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop() # in [sec]

def get_line_equation(point_a, point_b):
    """Compute line equation (ax+by+c=0).
    
    Args:
        point_a (vector,float): Coordinate of the first point
        point_b (vector,float): Coordinate of the second point
    
    Returns:
        vector,float: Coefficients [a,b,c] of the line 
    """
    a = point_b[1] - point_a[1] 
    b = point_a[0] - point_b[0]
    c = a*(point_a[0]) + b*(point_a[1])

    return [a, b, c]


def get_distance_from_line(line, point):
    """Calculate the shortest distance between the line and the point.
    
    Args:
        line (vector,float): Coefficients [a,b,c] of the line 
        point (vector,float): Coordinate of the point
    
    Returns:
        float: Shortest distance between the line and the point 
    """
    a, b, c = line
    x1, y1 = point
    num = a*x1 + b*y1 + c
    den = np.sqrt(a*a + b*b)
    if den == 0:
        den = 10e-5
        #raise('Division by zero!')

    return abs(num/den)

def project_point_to_line(point_a, point_b, point_c):
    """Compute the projected vector of the point c onto a line starting from the point a to the point b.
    
    Args:
        point_a (vector,float): Start point of the line
        point_b (vector,float): End point of the line
        point_c (vector,float): Point needs to be projected
    
    Returns:
        vector,float: Vector representing a line starting from the point a
    """
    v = point_b - point_a
    w = point_c - point_a
    w_on_v = np.dot(w,v)*v/np.dot(v,v)
    return w_on_v

def check_point_is_between_two_points(point_a, point_b, point_c):
    """Check whether the projecttion of the point c is lied between the points a and b.
    
    Args:
        point_a (vector,float): Start point of the line
        point_b (vector,float): End point of the line
        point_c (vector,float): Point needs to be projected
    
    Returns:
        binary and float: First return value tells whether the projected point c lies between the point a and b, 
        and the second output reports the distance between the first point (point a) and the project point c.  
    """
    is_between_between_two_points = False
    ac = project_point_to_line(point_a, point_b, point_c)
    bc = project_point_to_line(point_b, point_a, point_c)
    ab = point_b - point_a
    norm_ab = np.linalg.norm(ab)
    norm_ac = np.linalg.norm(ac)
    norm_bc = np.linalg.norm(bc)

    if max(norm_ac, norm_bc) <= norm_ab: # Point c is between point a and point b
        is_between_between_two_points = True
    
    return is_between_between_two_points, norm_ac


def kl_divergence_norm(x, p, q):
    mu1, sig1 = p 
    mu2, sig2 = q
    
    p = norm.pdf(x, mu1, sig1)
    q = norm.pdf(x, mu2, sig2)

    return kl_divergence(p, q)

def kl_divergence(p, q):

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))