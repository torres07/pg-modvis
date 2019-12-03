# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:39:29
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 19:44:25

import argparse
from space_invaders import SpaceInvaders

import config

parser = argparse.ArgumentParser(description='Train and test Q-networks on Space Invaders')

parser.add_argument('-m', '--mode', type=str, action='store', help='Please specify the mode you wish to run, either train or test (simulate)', required=True)
parser.add_argument('-l', '--load', type=str, action='store', help='Please specify the file you wish to load weights from (e.g. ddqn.h5)', required=False)
parser.add_argument('-n', '--ngames', type=int, action='store', help='Please specify the number of games that you want o simulate')
parser.add_argument('-f', '--usefps', action='store_true', help='Please specify if you want to use fps limit (60 by default)')

args = parser.parse_args()

if __name__ == '__main__':
	game_instance = SpaceInvaders()
	
	if args.mode == 'train':
		game_instance.train(config.TOT_FRAME)
		game_instance.model.save_network()

	if args.mode == 'test':
		game_instance.model.load_network(path=args.load)
		max_score, mean_score, std_score = game_instance.simulate(usefps=args.usefps, ngames=args.ngames)

		print('maximum score: {}'.format(max_score))
		print('mean score: {}'.format(mean_score))
		print('standard deviation of scores: {}'.format(std_score))
