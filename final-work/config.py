# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:05:15
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-11-30 15:34:46

DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 1000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 6
TAU = 0.01
LEARNING_RATE = 0.00001
# Number of frames to throw into network
NUM_FRAMES = 3