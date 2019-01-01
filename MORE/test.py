import numpy as np
import gym
import quanser_robots
from policy import *
# from sample import *
# from regression import * # , X
# from optimization import *
from MORE_iteration import *

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0] # = 5
action_dim = env.action_space.shape[0] # = 1
print("state_dim, action_dim =  ", state_dim, ", ", action_dim)
'''
    We assume that all degrees occur

    Policy is a polynomial p of degree 2, i.e.: a_2*x² + a_1*x + a_0
    a_i are parameter row-vectors of the form a_i = [a_1i, a_2i, ... , a_ni]
    where n = state_dim
    a_0 bias term
    p: state_dim -> action_dim
'''
degree = 2

'''
    Compute parameter for multivariate Gaussian to choose parameter
    Expectation value vector mu has dim(mu) = dim(a_2) + dim(a_1) + 1
    Set mu = 0
    Set covariance matrix Sigma
     - no covariance
     - variance for states with state_dim.low/high in (-inf, inf) should be almost zero
       to avoid output NaN if policy is computed (dirty soloution)
'''
#policy = NeuronalNetworkPolicy(state_dim, action_dim)
policy = PolynomialPolicy(state_dim, action_dim, degree)
#policy = DebugPolicy(state_dim, action_dim)

print("Number of model parameters: ", policy.get_number_of_parameters())


iterator = More(0.1, policy, env)

# setting reward 0 is always a bad idea..
thetas = iterator.iterate()
print("worked so far.")


# Call methods set theta to pass our policy the new params
