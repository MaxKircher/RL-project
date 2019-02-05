import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time

input = open("policies/cartpole_pol2.pkl", "rb")
#input = open("policies/my_policy_cartpole_cg.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

env = gym.make('CartpoleStabShort-v0')
s = env.reset()

for i in range(4000):
    env.render()
    # a = policy.get_action(s)
    # s, r, d, i = env.step(np.asarray(a))
    a = np.array(policy.get_action(s))
    print("a", a)
    print("s", s)
    s, r, done, info = env.step(a)
    if done or i%1000 == 0:
        s = env.reset()
    #time.sleep(.1)
