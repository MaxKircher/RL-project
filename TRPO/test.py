import numpy as np
import gym
import quanser_robots
from trpo import *
from policy import *
from sampling import *

env = gym.make('CartpoleStabShort-v0')
s0 = env.reset()
gamma = 0.99

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

policy = NN(s_dim, a_dim)
trpo = TRPO(env, gamma, policy)
iterations = 10 # recommanded 10 iterations on last page (above Appendix D)

for i in range(iterations):
    states, actions, Q = sample_sp(policy, s0, 1000, env, gamma)
    g = trpo.compute_loss_gradients(states, actions, Q)

    JM = np.matrix(trpo.compute_Jacobian(states))
    FIM = np.matrix(trpo.compute_FIM_mean())

    A = JM.T * FIM * JM # where A is the FIM w.r.t. to the Parameters theta see C
    print("dim(A): 2x2 < ", A.shape)

    # TODO: Als conjugate gradient schreiben
    s = np.linalg.lstsq(A, g.transpose(0,1))[0]
    beta = trpo.beta(0.01, np.matrix(s), A)

    theta_old = policy.get_parameter_as_tensor()

    policy = trpo.line_search(beta, 0.1, s, theta_old, states, actions, Q)

    theta_new = policy.get_parameter_as_tensor()

    # Correct forumla and does it work?
    delta = (theta_new - theta_old) / theta_old
    print("Iteration {} Relative change of parameter = ".format(i), delta)
