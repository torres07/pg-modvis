# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-10-21 16:40:52
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-11-18 15:41:54

class DeepQ(object):
	"""Summary
	
	Attributes:
	    model (keras.model): Description
	    target_model (keras.model): Description
	"""
	
	def __init__(self):
		"""Summary
		"""
		self.model = self.construct_q_network()
		self.target_model = self.construct_q_network()
		self.target_model.set_weights(self.model.get_weights())

	def construct_q_network(self):
		"""Summary
		
		Returns:
		    TYPE: Description
		"""
		model = Sequential()
		model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(NUM_ACTIONS))
		model.compile(loss='mse', optimizer=Adam(lr=0.00001))

		return model