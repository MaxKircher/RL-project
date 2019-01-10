import numpy as np
from matplotlib import pyplot as plt

# n = number of parameter
def plot(rewards, list_of_parameter, n):
    plt.figure()
    if n == 2:
        thetas = np.array(list_of_parameter)
        sc = plt.scatter(thetas[:,0], thetas[:,1], c=rewards)
        plt.colorbar(sc)
    else:
        thetas = get_thetas(list_of_parameter, n)
        plt.scatter(thetas, rewards)

    plt.show()


def get_thetas(lop, n):
    thetas = []
    for parameter in lop:
        thetas += [sum(parameter) / n]
    return np.array(thetas)
