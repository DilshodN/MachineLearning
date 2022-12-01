import math
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial import distance

def epanechnikov(dist):
    return np.where(np.abs(dist) <= 1.0, 3.0 / 4 * (1 - dist ** 2), 0.0)


def quadratic(dist):
    return np.where(np.abs(dist) <= 1.0, 15.0 / 16 * (1 - dist ** 2), 0.0)


def triangular(dist):
    return np.where(np.abs(dist) <= 1.0, 1 - abs(dist), 0.0)


def gauss(dist):
    return math.pow(2 * np.pi, -1.0 / 2) * np.exp(-1 / 2 * (dist ** 2))


def rect(dist):
    return np.where(np.abs(dist) <= 1.0, 1 / 2, 0.0)


class MPF:
    kernels = {
        "epanechnikov": epanechnikov,
        "quadratic": quadratic,
        "triangular": triangular,
        "gauss": gauss}
    
    
    def __init__(self,H = 5,kernel = "gauss", p = 2):
        self.H = H
        self.kernel = self.kernels[kernel]
        self.p = 2
                                
    def predict(self, X: np.array):
        test_X = np.copy(X)
        
        if len(test_X.shape) < 2:
            test_X = test_X[np.newaxis, :]
        
        diff = test_X[:, np.newaxis, :] - self.train_X[np.newaxis, :, :]
        distances = np.power(np.sum((diff ** self.p), -1), 1/self.p)
        weights = self.potentials * self.kernel(distances / self.H)
        
        classes = np.unique(self.train_y)
        table = np.zeros((test_X.shape[0], len(classes)))
        for c in classes:
            table[:, c] = np.sum(weights[:, self.train_y == c], axis=1)
        
        return np.argmax(table, axis=1)
    
    def predict_proba(self, X: np.array):
        test_X = np.copy(X)
        
        if len(test_X.shape) < 2:
            test_X = test_X[np.newaxis, :]
        
        diff = test_X[:, np.newaxis, :] - self.train_X[np.newaxis, :, :]
        distances = np.power(np.sum((diff ** self.p), -1), 1/self.p)
        weights = self.potentials * self.kernel(distances / self.H)
        
        classes = np.unique(self.train_y)
        table = np.zeros((test_X.shape[0], len(classes)))
        for c in classes:
            table[:, c] = np.sum(weights[:, self.train_y == c], axis=1)
        
        return np.array([table[:,0] / np.sum(table, axis=1), table[:,1] / np.sum(table, axis=1), table[:,2] / np.sum(table, axis=1)])
        
    def fit(self, X: np.array, y: np.array, epochs = 1):
        assert X.shape[0] == y.shape[0]
        self.train_X = np.copy(X)
        self.train_y = np.copy(y)
        self.potentials = np.zeros_like(y, dtype=int)
        
        for _ in range(epochs):
            for i in range(self.train_X.shape[0]):
                if self.predict(X[i]) != y[i]:
                    self.potentials[i] += 1
        
        self.zero_indexes = np.where(self.potentials == 0)[0]
        self.nonzero_indexes = np.nonzero(self.potentials)
        self.train_X = self.train_X[self.nonzero_indexes]
        self.train_y = self.train_y[self.nonzero_indexes]
        self.potentials = self.potentials[self.nonzero_indexes]