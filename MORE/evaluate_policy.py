import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time

###
from quanser_robots import GentlyTerminating
#env = GentlyTerminating(gym.make('CartpoleStabRR-v0'))
###

#input = open("policies/bb_rbf.pkl", "rb")
input = open("policies/CartpoleStabShort_scratch_poldeg2.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

env = gym.make('CartpoleStabShort-v0')


count = 0
rewards = 0
lof_rewards = []

for steps in range(50):
    s = env.reset()
    while count < 1000:
        # env.render()
        a = np.array(policy.get_action(s))
        s, r, done, info = env.step(a)
        rewards += r
        count += 1
        # time.sleep(0.1)
        if done == True:
            break
    lof_rewards += [rewards]
    rewards = 0
    count = 0

print("MORE: " , lof_rewards)
file = open("EvalSim/cartpole_poldeg2.npy", "wb")
np.save(file, lof_rewards)