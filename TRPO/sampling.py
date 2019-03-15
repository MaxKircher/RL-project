import numpy as np

def sample_sp(policy, s0, T, env, gamma):
    '''
    Samples values using the single path method (sp)
                see 5.1. Single Path

    :param policy: {NN} the policy that returns an action
    :param s0: {numpy ndarray} initial state that is the beginning of our sampling sequence
    :param T: {int} Number of samples
    :param env {TimeLimit} the environment
    :param gamma {float} discount factor

    :return:
     - states: {numpy ndarray} sampled states beginning with initial state s0
     - actions: {numpy ndarray} sampled actions by passing a state to our policy
     - Q: {numpy ndarray} state-action value function see page 2 above formula (1)
    '''
    s = s0
    states = [s0]
    actions = []
    rewards = []
    dones   = []
    for i in range(T):
        a = policy.choose_a(s)[0]
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
    states = np.array(states[:-1])
    actions = np.array(actions)
    Q = np.zeros(T + 1)

    # Compute discounted rewards:
    dones += [T-1]
    t0 = -1
    for tend in dones:
        for i in range(tend, t0, -1):
            Q[i] = gamma * Q[i + 1] + rewards[i]
        t0 = tend
    Q = Q[:-1]

    return states, actions, Q, sum(rewards)

