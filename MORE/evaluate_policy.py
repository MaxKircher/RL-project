import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time

input = open("policies/pendulum_nn.pkl", "rb")
#input = open("policies/my_policy_cartpole_cg.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

env = gym.make('CartpoleStabShort-v0')
s = env.reset()

for i in range(10000):
    env.render()
    # a = policy.get_action(s)
    # s, r, d, i = env.step(np.asarray(a))
    a = np.array(policy.get_action(s))
    print(a)
    s, r, done, info = env.step(a)
    if done:
        s = env.reset()
    time.sleep(.1)
