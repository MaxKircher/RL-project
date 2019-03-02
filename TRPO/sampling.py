import numpy as np

def sample_episode(policy, env):
    '''
    Samples a single episode
    :param policy: {NN} the policy that returns an action
    :param env: {TimeLimit} environment
    :return:
        - {numpy ndarray} sampled states
        - {numpy ndarray} sampled actions by passing a state to our policy
        - {numpy ndarray} sampled rewards
    '''
    s = env.reset()

    states = [s]
    actions = []
    rewards = []
    done = False
    while not done:
        a = policy.choose_action(s)[0]
        s, r, done, info = env.step(a)

        #todo necessary?
        #if type(s) is np.ndarray:
        #    s = tuple(s.reshape(-1))

        states  += [s]
        actions += [a]
        rewards += [r*100]

    return np.array(states), np.array(actions), np.array(rewards)


def sample_sp(env, policy, max_episodes):
    '''
    Samples values using the single path method (sp) see chapter 5.1.
    Creates lists with one array per episode, that contains the single steps.
    :param T: {int}  Number of samples
    :param env: {TimeLimit} environment
    :param policy: {NN} the policy that returns an action
    :param max_episodes:  {int}
    :return:
        - {list of numpy ndarray} sampled states
        - {list of numpy ndarray} sampled actions by passing a state to our policy
        - {list of numpy ndarray} sampled rewards
    '''
    states =  []
    actions = []
    rewards = []
    for i in range(max_episodes):
        s, a, r = sample_episode(policy, env)
        states  += [s]
        actions += [a]
        rewards += [r]
    return states, actions, rewards
