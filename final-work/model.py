# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-10-21 16:40:52
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 19:19:59

import numpy as np

from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

import config

class DeepQ(object):
	"""Definition of the Q-network and useful functions
	
	Attributes:
	    model (keras.model): Main Q-network, this network that chooses what actions are taken
	    target_model (keras.model): Target Q-network, this network that evaluates the actions
	"""
	
	def __init__(self):
		"""Summary
		"""
		self.model = self.construct_q_network()
		self.target_model = self.construct_q_network()
		self.target_model.set_weights(self.model.get_weights())

	def construct_q_network(self):
		"""Construct Q-network with convolutional layers and dense layers
		using RELU as activation function and MSE as loss function
		
		Returns:
		    keras.model: Q-network
		"""
		model = Sequential()
		model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, config.NUM_FRAMES)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(config.NUM_ACTIONS))
		
		model.compile(loss='mse', optimizer=Adam(lr=config.LEARNING_RATE))

		return model

	def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch):
		"""Train main Q-network with a batch = MINIBATCH_SIZE 
		and use target Q-network to evaluate and update Q-value
		
		Args:
		    s_batch (list): States batch
		    a_batch (list): Actions batch
		    r_batch (list): Rewards batch
		    d_batch (list): Dones batch
		    s2_batch (list): Next states (s') batch
		"""
		batch_size = s_batch.shape[0]
		targets = np.zeros((batch_size, config.NUM_ACTIONS))

		for i in range(batch_size):
			targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, config.NUM_FRAMES), batch_size = 1)
			fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, config.NUM_FRAMES), batch_size = 1)
			targets[i, a_batch[i]] = r_batch[i]
			if d_batch[i] == False:
				targets[i, a_batch[i]] += config.DECAY_RATE * np.max(fut_action)

		loss = self.model.train_on_batch(s_batch, targets)

	def target_train(self):
	    """Copy weights from main Q-network to target Q-network with a slightly change using parameter TAU
	    """
	    model_weights = self.model.get_weights()
	    target_model_weights = self.target_model.get_weights()
	    
	    for i in range(len(model_weights)):
	        target_model_weights[i] = config.TAU * model_weights[i] + (1 - config.TAU) * target_model_weights[i]
	    
	    self.target_model.set_weights(target_model_weights)		

	def predict_movement(self, data, epsilon):
		"""Summary
		
		Args:
		    data (np.array): Raw image
		    epsilon (float): Probability of perform a random action
		
		Returns:
		    (int, float): Optimal policy action, Q-value
		"""
		q_actions = self.model.predict(data.reshape(1, 84, 84, config.NUM_FRAMES), batch_size = 1)
		opt_policy = np.argmax(q_actions)
		rand_val = np.random.random()
		
		if rand_val < epsilon:
			opt_policy = np.random.randint(0, config.NUM_ACTIONS)
	
		return opt_policy, q_actions[0, opt_policy]

	def save_network(self, path='ddqn.h5'):
		"""Save model at specified path as h5 file
		
		Args:
		    path (str, optional): Path to save model
		"""
		self.model.save(path)
		print("Successfully saved network.")

	def load_network(self, path='ddqn.h5'):
		"""Load modeal at specified path, must be h5 file
		
		Args:
		    path (str, optional): Path to model
		"""
		self.model = load_model(path)