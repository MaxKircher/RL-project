import numpy as np
from matplotlib import pyplot as plt


def plot(rewards, list_of_parameter, n, b_history):
    '''
    If the dimension of the parameters is 2,
    this function plots the current memory samples and the means of all previous search distributions.
    :param rewards: {list} the rewards, corresponding to the parameters
    :param list_of_parameter: {list} the parameters, that shall be plotted
    :param n: {int} dimension of parameters
    :param b_history: {list} the search distribution means, that shall be plotted
    '''
    if n == 2:
        plt.figure()
        thetas = np.array(list_of_parameter)
        sc = plt.scatter(thetas[:,0], thetas[:,1], c=rewards)
        plt.colorbar(sc)
        b_history = np.array(b_history)
        plt.scatter(b_history[:,0], b_history[:,1], c="r", marker="x")
        plt.show(block=False)

