import numpy as np
import gym
import quanser_robots
from policy import *
from sample import *
from regression import * # , X
from optimization import *

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0] # = 5
action_dim = env.action_space.shape[0] # = 1

'''
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
mu = np.array(11*[0])
dev = 0.3*np.eye(11)
dev[4,4] = 1E-15
dev[5,5] = 1E-15
dev[9,9] = 1E-15
dev[10,10] = 1E-15

# Set quadratic policy
policy = POLICY(state_dim, action_dim, degree)

# Initalize sample generator
sample_generator = SAMPLE(env, policy, mu, dev)

# Generate samles for our policy
rewards, thetas = sample_generator.sample(10, 3)

'''
    Do linear regression
    rewards = X*beta

    Returns: beta
    # TODO: return R² (Maß für die Anpassungsgüte) -> see regression
'''
#X = X(thetas)
beta_hat = linear_regression(thetas, rewards)
print("beta: ", beta_hat.shape)
print("d = ", np.asarray(thetas).shape[1])
R, r, r0 = compute_quadratic_surrogate(beta_hat, np.asarray(thetas).shape[1])

opti = OPTIMIZATION(dev, mu, R, r, 1, 1)
x0 = np.ones(2) # starting point for etha and omega
g = opti.objective(x0)
print("g = ", g)
