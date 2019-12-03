# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:19:21
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 19:04:29

import numpy as np
import gym
import cv2
from replay_buffer import ReplayBuffer
from model import DeepQ
import time

import pickle

import config

class SpaceInvaders(object):
	"""SpaceInvaders (atari 2600) environment provided by gym
	
	Attributes:
	    env (gym.environment): Interface of gym that provides acess to environment of the game
	    model (keras.model): Q-network
	    process_buffer (list): Buffer that maintains last 3 frames
	    replay_buffer (ReplayBuffer): Buffer that stores experiences of the agent
	"""
	def __init__(self):
		"""Constructor for SpaceInvaders, here, the environment is initialized
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
		"""Pre process the raw images converting them into grayscale and resizing to (84, 84) pixels
		
		Returns:
		    np.array: Buffer of raw images
		"""
		black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 84)) for x in self.process_buffer]
		black_buffer = [x[0:84, :, np.newaxis] for x in black_buffer]
		
		return np.concatenate(black_buffer, axis=2)
		
	def train(self, num_frames):
		"""This function performs all steps to train the Q-network, during execution
		some logs are according to the number of observations that passed to provide a report of the progress.
		Note that the Q-network is retrained after the end of each game.
		Also, for each 10000 observations that passed, is saved a checkpoint of status of the model.
		
		At the end of training, a .txt file is saved with the report of all games, this report includes
		number of frames lived during the game and the score obtained. Also, a pickle file is saved with the same
		informations, this file will be used to plot the progress of the training.

		Args:
		    num_frames (int): Number of frames that Q-network will use to train
		"""
		rewards = []
		observation_num = 0
		curr_state = self.convert_process_buffer()
		epsilon = config.INITIAL_EPSILON
		alive_frame = 0
		total_reward = 0

		while observation_num < num_frames:
			if observation_num % 999 == 0:
				print(("Executing loop %d" % observation_num))

			# slowly decay the learning rate
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
				self.model.train(s_batch, a_batch, r_batch, d_batch, s2_batch)
				self.model.target_train()

			# save the network every 10000 iterations
			if observation_num % 10000 == 9999:
				print("Saving Network")
				self.model.save_network()

			alive_frame += 1
			observation_num += 1
		
		# files to provide future report
		with open('output.txt', 'w') as f:
			for i in rewards:
				f.write('{}, {}'.format(i[0], i[1]))
				f.write('\n')

		with open('rewards.pkl', 'wb') as f:
			pickle.dump(rewards, f)

	def simulate(self, ngames=1, usefps=False, fps=60.0):
		"""Simulate games using a pretrained model
		
		Args:
		    ngames (int, optional): Number of games to simulate
		    usefps (bool, optional): Use fps (frames per second) passed as parameter or set limit = max
		    fps (float, optional): Frames per second
		
		Returns:
		    (float, float, float): Maximum score obtained during games, mean score and standard deviation of scores 
		"""
		scores = []

		for i in range(ngames):
			done = False
			score = 0
			self.env.reset()
			self.env.render()

			while not done:
				state = self.convert_process_buffer()
				predict_movement = self.model.predict_movement(state, 0)[0]
				self.env.render()
				observation, reward, done, _ = self.env.step(predict_movement)
				score += reward
				self.process_buffer.append(observation)
				self.process_buffer = self.process_buffer[1:]
				if usefps:
					time.sleep(1.0/fps)

			scores.append(score)

		scores = np.array(scores)
		max_score = np.max(scores)
		mean_score = np.mean(scores)
		std_score = np.std(scores)

		return max_score, mean_score, std_score