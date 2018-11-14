import numpy as np
import gym
import quanser_robots

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
degree = 5


mu = np.matrix([26*[0]])
dev = np.matrix([26*[10]])




policy = POLICY(state_dim, action_dim, degree)
sample_generator = SAMPLE(env, policy, mu, dev)

result = sample_generator.sample(100, 3)
