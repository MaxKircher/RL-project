import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time


###
from quanser_robots import GentlyTerminating
env = GentlyTerminating(gym.make('BallBalancerRR-v1'))
###

#input = open("policies/gae.pkl", "rb")
input = open("policies/my_policy_BallBalancerSim_cont2_2.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

#env = gym.make('Pendulum-v2')
#env = gym.make('CartpoleStabShort-v0')


s = env.reset()
done = False
rewards = 0

while not done:
    env.render()
    a = policy.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
    s, r, done, info = env.step(a)
    rewards += r
    #time.sleep(0.1)

print("reward of episode: ", rewards)