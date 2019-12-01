# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-30 14:39:29
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-12-01 15:19:44

from space_invaders import SpaceInvaders
import config

if __name__ == '__main__':
    game_instance = SpaceInvaders()
    # game_instance.train(config.TOT_FRAME)
    # game_instance.model.save_network()

    game_instance.model.load_network(path='250.h5')
    game_instance.simulate(ngames=10)
