"""
Author: Willem Wybo
Date: 18/08/2015
Place: BBP, Geneva
"""

import math
import random


def poissonTrain(rate, tmax, tstart=4., seed=None):
    '''rate [kHz], tmax [ms], tstart [ms]'''
    if seed != None:
        random.seed(seed)
    spiketrain = []
    p = -math.log(1.0 - random.random()) / rate + tstart
    while p < tmax:
        spiketrain.append(p)
        p = (-math.log(1.0 - random.random()) / rate) + spiketrain[-1]
    return spiketrain
    

def modulatedPoissonTrain(rate, modrate, tmax, tstart=2.):
    '''rate [kHz], modrate [kHz], tmax [ms], tstart [ms]'''
    pass
