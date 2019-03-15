import numpy as np
import pickle
import gym
import quanser_robots
import torch

# Choose the policy you want to evaluate
policy_name = "CartpoleStabShort_scratch_50rbfs_rbf.pkl"
# Choose the environment, on that you want to evaluate:
env = gym.make('CartpoleStabShort-v0')


input = open("policies/" + policy_name, "rb")
data = pickle.load(input)
policy = data.get("policy")

lof_rewards = []

for steps in range(100):
    s = env.reset()
    rewards = 0
    for count in range(1000):
        #env.render()
        a = np.array(policy.get_action(s))
        s, r, done, info = env.step(a)
        rewards += r

        if done == True:
            break
    lof_rewards += [rewards]

print("MORE: " , lof_rewards)
file = open("EvalSim/cartpole_50rbf.npy", "wb")
np.save(file, lof_rewards)