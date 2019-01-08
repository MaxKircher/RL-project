import numpy as np
import gym
import quanser_robots
from trpo import *
from policy import *
from sampling import *
from conjugate_gradient import *
import pickle
from matplotlib import pyplot as plt

#plt.figure()
plt.show()
axes = plt.gca()
#plt.ion()
axes.set_xlim(0, 100)
rewards = np.array([]) # for plotting

#env = gym.make('CartpoleStabShort-v0')
env = gym.make('Pendulum-v2')
s0 = tuple(env.reset())
gamma = 0.9999

delta = 0.01 # KL threshold in linesearch

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

policy = NN(s_dim, a_dim)
trpo = TRPO(env, gamma, policy)

# recommanded 10 iterations on last page (above Appendix D)
cg = ConjugateGradient(10)

iterations = 100
# Table 2 -> min 50.000
num_steps = 10000
for i in range(iterations):
    print("Iteration ", i, ":")

    states, actions, Q, r = sample_sp(policy, s0, num_steps, env, gamma)
    rewards = np.append(rewards, r) # for plotting

    g = trpo.compute_objective_gradients(states, actions, Q).detach().numpy().T

    subsampled_states = states[0::10] #get every tenth state (see above App D)

    JM = np.matrix(trpo.compute_Jacobian(subsampled_states))
    FIM = np.matrix(trpo.compute_FIM_mean())

    A = JM.T * FIM * JM # where A is the FIM w.r.t. to the Parameters theta see C

    s = np.linalg.lstsq(A, g.transpose(0,1), rcond=None)[0]
    # TODO: Startwert? g, should be kind of similar to s
    #s = cg.cg(g, JM, FIM, g)

    #print("cg: ", s_cg.T)
    #print("lstsq: ", s.T)

    beta = trpo.beta(0.01, np.matrix(s), JM, FIM)

    theta_old = policy.get_parameter_as_tensor().detach()

    policy = trpo.line_search(beta, delta, s, theta_old, states, actions, Q)
    trpo.policy = policy

    # Printing:
    #theta_new = policy.get_parameter_as_tensor()
    #print_delta = (theta_new - theta_old) / theta_old
    #print("Iteration {} Relative change of parameter = ".format(i), print_delta)


    # Save in file
    dict = {"policy": policy}
    with open("my_policy2.pkl", "wb") as output:
        pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

    # Plotting
    plt.plot(range(i+1), rewards, c='b')
    plt.draw()
    plt.pause(1e-17)
