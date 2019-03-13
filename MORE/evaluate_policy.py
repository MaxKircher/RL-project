import numpy as np
import pickle
import gym
import quanser_robots
import torch
import time

input = open("policies/bb_rbf.pkl", "rb")
#input = open("policies/my_policy_cartpole_cg.pkl", "rb")
data = pickle.load(input)
#policy = data.get("policy")
policy = data.get("policy")

env = gym.make('BallBalancerSim-v0')
s = env.reset()
done = False
rewards = 0

while not done:
    env.render()
    # a = policy.get_action(s)
    # s, r, d, i = env.step(np.asarray(a))
    a = np.array(policy.get_action(s))
    print("a", a)
    print("s", s)
    s, r, done, info = env.step(a)
    #time.sleep(.1)
    rewards += r

print("reward of episode: ", rewards)