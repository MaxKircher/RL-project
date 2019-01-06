import numpy as np
from matplotlib import pyplot as plt

def plot(rewards, x):
    plt.figure()
    plt.scatter(x, rewards)

    plt.show()
