import numpy as np
from matplotlib import pyplot as plt

# n = number of parameter
def plot(rewards, list_of_parameter, n):
    #rewards = np.array(rewards)
    plt.figure()
    thetas = get_thetas(list_of_parameter, n)
    sc = plt.scatter(thetas[:,0], thetas[:,1], c=rewards)
    plt.colorbar(sc)
    #print("Shapes:\n", thetas[:,0].shape, thetas[:,1].shape, rewards.shape)
    #plt.contour(thetas[:,0], thetas[:,1], rewards)

    plt.show()

def get_thetas(lop, n):
    thetas = []
    for parameter in lop:
        thetas += [parameter / n]
    return np.array(thetas)
