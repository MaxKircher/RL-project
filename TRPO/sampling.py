import numpy as np

'''
Samples values using the single path method (sp) see chapter 5.1

Parameter:
 - policy: {NN} the policy that returns an action
 - T:      {int}  Number of samples
 - env:    {TimeLimit} environment
 - gamma:  {float} discount factor

Returns:
 - {numpy ndarray} sampled states
 - {numpy ndarray} sampled actions by passing a state to our policy
 - {numpy ndarray} state-action value function (Monte Carlo estimate)
 - {float} summed reward over all samples
'''
def sample_sp(policy, T, env, gamma):
    s = tuple(env.reset())

    states = [s]
    actions = []
    rewards = []
    dones   = []
    for i in range(T):
        a = policy.choose_action(s)[0]
        s, r, done, info = env.step(a)
        if type(s) is np.ndarray:
            s = tuple(s.reshape(-1))

        if done:
            s = tuple(env.reset())
            dones += [i]

        states  += [s]
        actions += [a]
        rewards += [100*r]

    states = np.array(states[:-1])
    actions = np.array(actions)
    Q = np.zeros(T + 1)

    dones += [T-1]
    t0 = -1
    # Go through all episodes:
    for t_end in dones:
        # Compute Q values for one episode
        for i in range(t_end, t0, -1):
            Q[i] = gamma * Q[i + 1] + rewards[i]
        t0 = t_end
    Q = Q[:-1]

    return states, actions, Q, sum(rewards)
