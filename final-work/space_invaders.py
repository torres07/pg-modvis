# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:19:21
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 10:14:12

import numpy as np
import gym
import cv2
from replay_buffer import ReplayBuffer
from model import DeepQ

import pickle

import config

class SpaceInvaders(object):
	"""docstring for SpaceInvaders
	
	Attributes:
		env (gym.environment): Interface of gym that provides acess to environment of the game
		process_buffer (list): Buffer that maintains last 3 frames
	"""
	def __init__(self):
		"""Summary
		"""
		self.env = gym.make('SpaceInvaders-v0')
		self.env.reset()
		self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
		self.model = DeepQ()

		s1, r1, _, _ = self.env.step(0)
		s2, r2, _, _ = self.env.step(0)
		s3, r3, _, _ = self.env.step(0)

		self.process_buffer = [s1, s2, s3]

	def convert_process_buffer(self):
		black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.process_buffer]
		black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]
		
		return np.concatenate(black_buffer, axis=2)
		
	def train(self, num_frames):
		rewards = []
		observation_num = 0
		curr_state = self.convert_process_buffer()
		epsilon = config.INITIAL_EPSILON
		alive_frame = 0
		total_reward = 0

		while observation_num < num_frames:
			if observation_num % 999 == 0:
				print(("Executing loop %d" % observation_num))

			# Slowly decay the learning rate
			if epsilon > config.FINAL_EPSILON:
				epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / config.EPSILON_DECAY

			initial_state = self.convert_process_buffer()
			self.process_buffer = []

			predict_movement, predict_q_value = self.model.predict_movement(curr_state, epsilon)

			reward, done = 0, False
			for i in range(config.NUM_FRAMES):
				temp_observation, temp_reward, temp_done, _ = self.env.step(predict_movement)
				reward += temp_reward
				self.process_buffer.append(temp_observation)
				done = done | temp_done

			# if observation_num % 10 == 0:
			# 	print("We predicted a q value of ", predict_q_value)

			if done:
				print("Lived with maximum time ", alive_frame)
				print("Earned a total of reward equal to ", total_reward)
				rewards.append((total_reward, alive_frame))
				self.env.reset()
				alive_frame = 0
				total_reward = 0

			new_state = self.convert_process_buffer()
			self.replay_buffer.add(initial_state, predict_movement, reward, done, new_state)
			total_reward += reward

			if self.replay_buffer.size() > config.MIN_OBSERVATION:
				s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(config.MINIBATCH_SIZE)
				self.model.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
				self.model.target_train()

			# Save the network every 100000 iterations
			if observation_num % 10000 == 9999:
				print("Saving Network")
				self.model.save_network()

			alive_frame += 1
			observation_num += 1
		
		with open('output.txt', 'w') as f:
			for i in rewards:
				f.write('{}, {}'.format(i[0], i[1]))
				f.write('\n')

		with open('rewards.pkl', 'wb') as f:
			pickle.dump(rewards, f)