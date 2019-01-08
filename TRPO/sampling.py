import numpy as np

'''
Samples values using the single path method (sp)

Parameter:
 - policy:  the policy that returns an action
 - s0:      initial state that is the beginning of our sampling sequence
            see 5.1. Single Path
- T:        Number of samples

Returns:
 - states:  sampled states beginning with initial state s0
 - actions: sampled actions by passing a state to our policy
 - Q:       state-action value function see page 2 above formula (1)
'''
def sample_sp(policy, s0, T, env, gamma):
    s = s0
    states = [s0]
    actions = []
    rewards = []
    dones   = []
    for i in range(T):
        a = policy.choose_a(s)
        s, r, done, info = env.step(a)
        if type(s) is np.ndarray:
            s = tuple(s.reshape(-1))

        if done:
            s = tuple(env.reset())
            dones += [i]

        states  += [s]
        actions += [a]
        rewards += [r]

    # Make an array from the lists states and actions
    states = np.array(states)
    actions = np.array(actions)
    Q = np.zeros(T + 1)

    dones += [T-1]
    t0 = -1
    for tend in dones:
        for i in range(tend, t0, -1):
            Q[i] = gamma * Q[i + 1] + rewards[i]
        t0 = tend

    # print("s_mean: ", states.mean(0))
    # print("s_std: ", states.std(0))
    # print("a_mean: ", actions.mean(0))
    # print("a_std: ", actions.std(0))
    return states, actions, Q




# Evaluated once for pendulum:
# s_mean =  np.array([-0.47201  -0.053921])
# s_std =  np.array([2.759836 1.75064 ])
# a_mean =  np.array([0.390378])
# a_std =  np.array([1.739343])
#
# def normalize(states, actions):
#     states = (states - s_mean) / s_std
#     actions = (actions - a_mean) / a_std
#     return states, actions
