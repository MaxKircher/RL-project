import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time

input = open("policies/my_policy_pendulum_cg_cont_800.pkl", "rb")
#input = open("policies/my_policy_cartpole_cg.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy_cartpole_cg")

env = gym.make('Pendulum-v2')
#env = gym.make('CartpoleStabShort-v0')
s = env.reset()

for i in range(1000):
    env.render()
    a = policy.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
    s, r, done, info = env.step(a)
    if done:
        s = env.reset()
    time.sleep(0.1)
