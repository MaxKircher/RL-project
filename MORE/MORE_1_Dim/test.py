import numpy as np
import gym
import quanser_robots
from policy import *
from MORE_iteration import *

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0] # = 5
action_dim = env.action_space.shape[0] # = 1

policy = DebugPolicy(state_dim, action_dim)

print("Number of model parameters: ", policy.get_number_of_parameters())
iterator = More(0.1, policy, env)

thetas = iterator.iterate()
print("worked so far.")
