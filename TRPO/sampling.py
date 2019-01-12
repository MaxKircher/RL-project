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



 ---> cf.: https://git.ias.informatik.tu-darmstadt.de/quanser/clients/tree/master/quanser_robots/cartpole

    import gym
    from quanser_robots import GentlyTerminating
    env = GentlyTerminating(gym.make('CartpoleSwingRR-v0'))
    ctrl = ...  # some function f: s -> a
    obs = env.reset()
    done = False
    while not done:
        act = ctrl(obs)
        obs, rwd, done, info = env.step(act)

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

    Q = (Q - Q.mean()) / Q.std()

    return states, actions, Q, sum(rewards)
