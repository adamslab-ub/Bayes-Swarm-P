#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/25/2020 """

import numpy as np 
import pickle
from matplotlib import pyplot as plt

from BayesSwarm.source import Source
from postprocessing.util.plot import *

import pandas as pd
import seaborn as sns

output_dir = "postprocessing"
source_id = 10
if source_id == 2:
    levels = 10
elif source_id == 7:
    levels = 10
elif source_id == 10:
    levels = 20
file_name = output_dir + "/source_signal_" + str(source_id)

source = Source(source_id)
X1, X2, Y = source.get_data_for_plot()

fig, axes = latexify()

colors=[payamPalette_b[0],payamPalette_b[2],payamPalette_b[6]]

plt.contourf(X1, X2, Y, levels, cmap='viridis')
plt.contour(X1, X2, Y, levels, colors="k", linewidths=1)
n_robots = 25
r_movement = 100
dtheta = 0.5 * np.pi / (n_robots + 1)
for i in range(n_robots):
    x = -120 + r_movement * np.cos((i + 1) * dtheta)
    y = -120 + r_movement * np.sin((i + 1) * dtheta)
    plt.plot(x, y, 'o')
for i in range(n_robots):
    x = 120 - r_movement * np.cos((i + 1) * dtheta)
    y = 120 - r_movement * np.sin((i + 1) * dtheta)
    plt.plot(x, y, 's')
for i in range(n_robots):
    x = -120 + r_movement * np.cos((i + 1) * dtheta)
    y = 120 - r_movement * np.sin((i + 1) * dtheta)
    plt.plot(x, y, 's')
for i in range(n_robots):
    x = 120 - r_movement * np.cos((i + 1) * dtheta)
    y = -120 + r_movement * np.sin((i + 1) * dtheta)
    plt.plot(x, y, 's')
plt.ylabel("x [m]")
plt.xlabel("y [m]")
plt.axis("square")
fontsize = 10
    
plt.savefig(file_name + '.pdf', format='pdf', dpi=300, bbox_inches='tight')    
plt.savefig(file_name + '.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
