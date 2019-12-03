# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:05:15
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 16:17:43

DECAY_RATE = 0.99 # parameter to find optimal policy (how much Q-value will be updated)
BUFFER_SIZE = 16384 # number of experiences that can be stored on the replay buffer
MINIBATCH_SIZE = 64 # size of the batch used to re-train the network
TOT_FRAME = 1000 # number of frames (observations) that the agent will use during training
MIN_OBSERVATION = 100 # minimum of observations seen before starting training
EPSILON_DECAY = 1000000 # used to slowly decay learning rate
INITIAL_EPSILON = 0.1 # initial chance of peform a random action
FINAL_EPSILON = 0.01 # final chance of peform a random action
TAU = 0.01 # parameter to make a subtle change in the network weight copy (main -> target)
LEARNING_RATE = 0.00001 # learning rate used on optmizer
NUM_ACTIONS = 6 # possible actions that the agent can peform: {fire, move right, move left, fire and move right, fire and move left, no operation}
NUM_FRAMES = 3 # number of frames to throw into network
