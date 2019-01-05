import numpy as np
from matplotlib import pyplot as plt
import pickle
import gym
import quanser_robots
import torch

#with open("policy.pkl", "rb") as input:
input = open("my_policy2.pkl", "rb")
data = pickle.load(input)
policy = data.get("policy")

env = gym.make('Pendulum-v2')
s0 = env.reset()

low = env.observation_space.low
high = env.observation_space.high
step = [0.01,0.05]
X = np.arange(low[0],high[0],step[0])
Y = np.arange(low[1],high[1],step[1])
states = np.asarray([(x,y) for x in X for y in Y])


print(states)
a = policy.model(torch.tensor(states, dtype=torch.float)).detach().numpy().reshape(X.shape[0], Y.shape[0])
#a = policy.choose_a(s0)

#x = states[:,0]
#y = states[:,1]
print("shapes: ", X.shape, Y.shape, a.shape)

fig, ax = plt.subplots()
CS = plt.contour(a,15)
ax.clabel(CS, inline=1, fontsize=10)
plt.show()
