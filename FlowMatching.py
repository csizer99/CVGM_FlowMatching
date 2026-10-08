import torch
import numpy as np

class Flow():
    def __init__(self):
        self.points = self.generatePoints()


    def generatePoints(self, mu = 50, sigma = 5, num_points = 200):
        blob1 = np.random.normal([20,20], 5, (100, 2))
        blob2 = np.random.normal([80, 80], 5, (100, 2))
        return np.concatenate([blob1, blob2], axis=0)

    def getPoints(self):
        return self.points




