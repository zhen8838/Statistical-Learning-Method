import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel, Dict
import pprint


class DecisionTreeID3(BaseModel):
  def __init__(self, thresh):
    super().__init__()
    self.thresh = thresh

  @staticmethod
  def entropy(y):
    N = len(y)
    count = np.array([len(y[y == k]) for k in np.unique(y)])
    entro = -np.sum((count / N) * (np.log2(count / N)))
    return entro

  @staticmethod
  def cond_entropy(X, y, fidx):
    N = len(y)
    cond_X = X[:, fidx]
    tmp_entro = []
    for val in set(cond_X):
      tmp_y = y[np.where(cond_X == val)]
      tmp_entro.append(len(tmp_y) / N * DecisionTreeID3.entropy(tmp_y))
    cond_entro = sum(tmp_entro)
    return cond_entro

  @staticmethod
  def makemap(arr):
    tag = np.unique(arr)
    tag_map = Dict(zip(tag, range(len(tag))))
    return tag_map

  @staticmethod
  def info_gain(X, Y, fidx):
    return DecisionTreeID3.entropy(Y) - DecisionTreeID3.cond_entropy(X, Y, fidx)

  @staticmethod
  def info_gain_ratio(X, Y, fidx):
    return (DecisionTreeID3.entropy(Y) - DecisionTreeID3.cond_entropy(X, Y, fidx)) / DecisionTreeID3.cond_entropy(X, Y, fidx)

  @staticmethod
  def get_best_fidx(X, Y):
    assert X.ndim == 2
    info_gain = [DecisionTreeID3.info_gain(X, Y, fidx)
                 for fidx in range(X.shape[-1])]
    return np.argmax(info_gain), np.max(info_gain)

  @staticmethod
  def max_instance(Y):
    unique, counts = np.unique(Y, return_counts=True)
    max_idx = np.argmax(counts)
    return unique[max_idx]

  def fit(self, X, Y, labels):
    labels = labels.copy()
    M, N = X.shape
    if len(np.unique(Y)) == 1:
      return Y[0]

    if N <= 1:
      return self.max_instance(Y)

    bestSplit, best_gain = self.get_best_fidx(X, Y)
    if best_gain < self.thresh:
      return self.max_instance(Y)

    bestFeaLable = labels[bestSplit]
    Tree = {bestFeaLable: {}}
    del labels[bestSplit]
    
    feaVals = np.unique(X[:, bestSplit])
    for val in feaVals:
      idx = np.where(X[:, bestSplit] == val)
      # sub_X = X[idx]
      sub_X = np.delete(X, bestSplit, axis=-1)[idx]
      sub_y = Y[idx]
      sub_labels = labels
      Tree[bestFeaLable][val] = self.fit(sub_X, sub_y, sub_labels)

    return Tree


if __name__ == "__main__":

  data = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 2, 1],
                   [1, 0, 1, 2, 0],
                   [1, 0, 0, 1, 0],
                   [1, 1, 1, 1, 1],
                   [2, 1, 1, 1, 1],
                   [2, 1, 1, 2, 1],
                   [2, 0, 0, 2, 0],
                   [2, 1, 0, 3, 0],
                   [2, 1, 0, 3, 0],
                   [3, 1, 0, 3, 0],
                   [3, 1, 0, 2, 0],
                   [3, 0, 1, 2, 0],
                   [3, 0, 1, 3, 0],
                   [3, 1, 1, 1, 1]])
  X, Y = data[:, :-1], data[:, -1]

  id3 = DecisionTreeID3(thresh=0.1)

  id3.get_best_fidx(X, Y)
  tree = id3.fit(X, Y, ['年龄', '有工作', '有自己的房子', '信贷情况'])
  pprint.pprint(tree)
