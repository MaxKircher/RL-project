import numpy as np
import gym
import quanser_robots
from policy import *
from MORE_iteration import *

env = gym.make('BallBalancerSim-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


#policy = LinearPolynomial(state_dim, action_dim, 2)
policy = LinearRBF(state_dim, action_dim, 100)

#policy = Rastrigin(state_dim, action_dim)
#policy = Rosenbrock(state_dim, action_dim)

print("Number of model parameters: ", policy.get_number_of_parameters())

# The convergence criterion:
bound = 1

N_per_theta, number_of_thetas, memory_size = 1,50,1000 # For policies
# N_per_theta, number_of_thetas, memory_size = 1, 1000, 1000 # For Debug

iterator = MORE(policy, env, N_per_theta, number_of_thetas, memory_size)
thetas = iterator.iterate(bound) # Initial etha / omega can be set here

print("MORE converged to the above soloution.")
