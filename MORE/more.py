#for the algorithm
import numpy as np
import gym
import quanser_robots


env = gym.make('CartpoleStabShort-v0')
s0 = env.reset()
