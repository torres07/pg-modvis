# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-10-21 15:22:15
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 16:01:55

import random
import numpy as np
from collections import deque

class ReplayBuffer(object):
	"""Buffer that stores experiences of the agent
	
	Attributes:
	    buffer (collection.deque): buffer of experiences
	    buffer_size (int): size of buffer
	"""
	def __init__(self, buffer_size):
		"""ReplayBuffer constructor
		
		Args:
		    buffer_size (int): Size of buffer
		"""
		self.buffer_size = buffer_size
		self.buffer = deque()

	def add(self, s, a, r, d, s_):
		"""Add a new experience in buffer
		
		Args:
		    s (TYPE): State
		    a (int): Action
		    r (float): Reward
		    d (bool): End
		    s_ (TYPE): Next state
		"""
		experience = (s, a, r, d, s_)
		
		if self.size() < self.buffer_size:
			self.buffer.append(experience)
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def sample(self, batch_size):
		"""Return samples of experience size of batch_size
		
		Args:
		    batch_size (int): Batch size
		
		Returns:
		    tuple: Batch of states, actions, rewards, ends and new_states
		"""
		batch = list()

		if self.size() < batch_size:
			batch = random.sample(self.buffer, self.size())
		else:
			batch = random.sample(self.buffer, batch_size)

		s_batch, a_batch, r_batch, d_batch, s__batch = list(map(np.array, list(zip(*batch))))

		return s_batch, a_batch, r_batch, d_batch, s__batch

	def size(self):
		"""Return size of replay buffer
		
		Returns:
		    int: Size of buffer
		"""
		return len(self.buffer)

	def clear(self):
		"""Clear buffer
		"""
		self.buffer.clear()