import numpy as np
import pickle
import gym
import quanser_robots
import torch

# Choose the policy you want to evaluate:
policy_name = "my_policy_cartpole_new_quanser.pkl"
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
        a = policy.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
        s, r, done, info = env.step(a)
        rewards += r

        if done == True:
            break
    lof_rewards += [rewards]

print("TRPO", lof_rewards)
file = open("EvalSim/TRPO_eval_sim_cartpole.npy", "wb")
np.save(file, lof_rewards)