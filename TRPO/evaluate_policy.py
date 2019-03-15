import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time


###
from quanser_robots import GentlyTerminating
#env = GentlyTerminating(gym.make('BallBalancerRR-v1'))
###

#input = open("policies/gae.pkl", "rb")
input = open("policies/my_policy_cartpole_new_quanser.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

#env = gym.make('Pendulum-v2')
env = gym.make('CartpoleStabShort-v0')



count = 0
rewards = 0
lof_rewards = []

for steps in range(50):
    s = env.reset()
    while count < 1000:
        #env.render()
        a = policy.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
        s, r, done, info = env.step(a)
        rewards += r
        count += 1
        #time.sleep(0.1)
        if done == True:
            break
    lof_rewards += [rewards]
    rewards = 0
    count = 0

print("TRPO", lof_rewards)
file = open("EvalSim/TRPO_eval_sim_cartpole.npy", "wb")
np.save(file, lof_rewards)