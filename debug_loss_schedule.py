# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:26:33 2020

@author: chenlin
"""

import numpy as np
loss_schedule = {'0': [0.01, 1.0, 0.01], '5': [0.1, 1.0, 0.01], '6':  [0.2, 1.0, 0.01], '7':[0.5, 1.0, 0.01], '8':[1.0, 0.5 ,0.001] }
tilt_schedule = {'0': 4.0, '1': 4.5, '2':  4.8, '3':5.3, '4':5.8 }
for epoch in np.arange(0,20):
    
    ep1 = epoch
    
    while str(ep1) not in tilt_schedule.keys():
        ep1 -=1
        if ep1 < 0:
            break
    max_tilt = tilt_schedule[str(ep1)]
    
    ep1 = epoch
    while str(ep1) not in loss_schedule.keys():
        ep1 -=1
        if ep1 < 0:
            break
    loss_weights = loss_schedule[str(ep1)]
    weight_ori = loss_weights[0]
    weight_stretch = loss_weights[1]
    weight_skew = loss_weights[2]
    print(weight_ori, weight_stretch, weight_skew)