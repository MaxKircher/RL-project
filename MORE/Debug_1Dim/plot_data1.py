import numpy as np
from matplotlib import pyplot as plt

# n = number of parameter
def plot(rewards, x):
    plt.figure()
    plt.scatter(x, rewards)

    plt.show()
