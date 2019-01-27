import numpy as np
from matplotlib import pyplot as plt

# n = number of parameter
def plot(rewards, list_of_parameter, n, b_history):
    if n == 2:
        plt.figure()
        thetas = np.array(list_of_parameter)
        sc = plt.scatter(thetas[:,0], thetas[:,1], c=rewards)
        plt.colorbar(sc)
        b_history = np.array(b_history)
        plt.scatter(b_history[:,0], b_history[:,1], c="r", marker="x")
        plt.show(block=False)
    # else:
    #     thetas = get_thetas(list_of_parameter, n)
    #     plt.scatter(thetas, rewards)


def get_thetas(lop, n):
    thetas = []
    for parameter in lop:
        thetas += [sum(parameter) / n]
    return np.array(thetas)
