import numpy as np
import gym
import quanser_robots
from policy import *
from sample import *
from regression import X, linear_regression

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
degree = 2


mu = np.array(11*[0])
dev = 0.3*np.eye(11)
dev[4,4] = 1E-15
dev[5,5] = 1E-15
dev[9,9] = 1E-15
dev[10,10] = 1E-15


policy = POLICY(state_dim, action_dim, degree)
sample_generator = SAMPLE(env, policy, mu, dev)

rewards, thetas = sample_generator.sample(10, 3)

# do linear regression
X = X(thetas)
beta = linear_regression(X, rewards)
print("beta: ", beta.shape)
