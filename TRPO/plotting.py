from matplotlib import pyplot as plt
import numpy as np


class LearningCurvePlotter(object):
    def __init__(self, number_of_iterations, filename):
        self.rewards = []
        self.vars = []
        self.filename = filename

        plt.show()
        self.axes = plt.gca()
        self.axes.set_xlim(0, number_of_iterations)


    def update(self, reward, variance):
        self.rewards += [reward]
        self.vars += [variance]
        plt.plot(range(len(self.rewards)), self.rewards, c='b')
        plt.draw()
        plt.pause(1e-17)

        if self.filename is not None:
            plt.savefig("snapshots/%s.png" %self.filename)
            file = open("snapshots/%s.npy" %self.filename, "wb")
            np.save(file, [self.rewards, self.vars])

