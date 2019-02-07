import numpy as np
import gym
import quanser_robots
from trpo import line_search
from policy import NN
from sampling import sample_sp
import pickle
from matplotlib import pyplot as plt
from quanser_robots import GentlyTerminating

plt.show()
axes = plt.gca()

iterations = 800

axes.set_xlim(0, iterations)
rewards = np.array([]) # for plotting

env = gym.make('Qube-v0')
#env = gym.make('Pendulum-v2')

#env = GentlyTerminating(gym.make('BallBalancerRR-v0'))

gamma = 0.99

delta = 0.1 # KL threshold in linesearch

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

policy = NN(s_dim, a_dim)

#input = open("policies/debugging3.pkl", "rb")
#data = pickle.load(input)
#policy = data.get("policy")


# Table 2 -> min 50.000
num_steps = 2000
for i in range(iterations):
    print("Iteration ", i, ":")

    states, actions, Q, r = sample_sp(policy, num_steps, env, gamma)

    rewards = np.append(rewards, r) # for plotting

    policy = line_search(delta, states, actions, Q, policy)
    print("STD: ", policy.model.log_std.exp())

    # Save in file
    #dict = {"policy": policy}
    #with open("policies/low_var.pkl", "wb") as output:
    #    pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

    # Plotting
    plt.plot(range(i+1), rewards, c='b')
    plt.draw()
    plt.pause(1e-17)
    #plt.savefig("snapshots/low_var.png")
