import numpy as np
from matplotlib import pyplot as plt

# n = number of parameter
def plot(rewards, list_of_parameter, n):
    plt.figure()
    thetas = get_thetas(list_of_parameter, n)
    plt.scatter(thetas, rewards)

    plt.show()

def get_thetas(lop, n):
    thetas = []
    #print("Length, list_of_parameter (lop): ", len(lop))
    for parameter in lop:
        #print("parameter: ", parameter)
        thetas += [sum(parameter) / n]
    return thetas
