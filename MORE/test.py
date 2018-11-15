import numpy as np
import gym
import quanser_robots
from policy import *
from sample import *

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
degree = 2


mu = np.array(11*[0])
dev = 0.3*np.eye(11)
dev[4,4] = 1E-5
dev[5,5] = 1E-5
dev[9,9] = 1E-5
dev[10,10] = 1E-5


policy = POLICY(state_dim, action_dim, degree)
sample_generator = SAMPLE(env, policy, mu, dev)

result = sample_generator.sample(10, 3)
print(result)
