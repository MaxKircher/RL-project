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

iterations = 800

axes.set_xlim(0, iterations)
rewards = np.array([]) # for plotting

env = gym.make('Qube-v0')
#env = gym.make('Pendulum-v2')
s0 = tuple(env.reset())
gamma = 0.9999

delta = 0.01 # KL threshold in linesearch

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

# policy = NN(s_dim, a_dim)

input = open("policies/my_policy_qube_cont_bb.pkl", "rb")
data = pickle.load(input)
policy = data.get("policy")

trpo = TRPO(env, gamma, policy)

# recommanded 10 iterations on last page (above Appendix D)
cg = ConjugateGradient(10)
# Table 2 -> min 50.000
num_steps = 500
for i in range(iterations):
    print("Iteration ", i, ":")

    if env.__str__() == '<TimeLimit<BallBalancerSim<BallBalancerSim-v0>>>':
        states, actions, Q, r = sample_sp_bb(policy, s0, num_steps, env, gamma)
    else:
        states, actions, Q, r = sample_sp(policy, s0, num_steps, env, gamma)

    rewards = np.append(rewards, r) # for plotting

    g = trpo.compute_objective_gradients(states, actions, Q).detach().numpy().T

    subsampled_states = states[0::10] #get every tenth state (see above App D)

    JMs = trpo.compute_Jacobians(subsampled_states)
    FIM = np.matrix(trpo.compute_FIM_mean())

    A = np.zeros((JMs[0].shape[1],JMs[0].shape[1]))
    for j in range(len(JMs)):
        A_x = np.matrix(JMs[j]).T @ FIM @ np.matrix(JMs[j]) # where A is the FIM w.r.t. to the Parameters theta see C
        A += A_x

    A_avg = A / len(JMs)
    print("Rank(A_avg) = ", np.linalg.matrix_rank(A_avg))
    print("A_avg.shape = ", A_avg.shape)
    # s = np.linalg.lstsq(A_avg, g.transpose(0,1), rcond=None)[0]
    # TODO: Startwert? g, should be kind of similar to s
    s_cg = cg.cg(g, JMs, FIM, g)

    # print("cg: ", s_cg.T)
    # print("lstsq: ", s.T)
    # print("cg - lstsq: ", s_cg.T - s.T)

    # beta = trpo.beta(0.01, np.matrix(s), A_avg)
    beta_cg = trpo.beta(0.01, np.matrix(s_cg), A_avg)

    # print("beta: ", beta)
    # print("beta_cg: ", beta_cg)
    # print("beta - beta_cg: ", beta - beta_cg)

    theta_old = policy.get_parameter_as_tensor().detach()

    policy = trpo.line_search(beta_cg, delta, s_cg, theta_old, states, actions, Q)
    trpo.policy = policy

    # Printing:
    #theta_new = policy.get_parameter_as_tensor()
    #print_delta = (theta_new - theta_old) / theta_old
    #print("Iteration {} Relative change of parameter = ".format(i), print_delta)


    # Save in file
    dict = {"policy": policy}
    with open("policies/my_policy_qube_cont_bb.pkl", "wb") as output:
        pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

    # Plotting
    plt.plot(range(i+1), rewards, c='b')
    plt.draw()
    plt.pause(1e-17)
    plt.savefig("snapshots/my_policy_qube_cont_4.png")
