import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Hierarchical(BaseModel):
  def __init__(self, p: float = 2):
    super().__init__()
    self.p = p
    self.history = []

  def distance(self, x: np.ndarray, y: np.ndarray):
    dis = np.power(np.sum(np.power(np.abs((x - y)), self.p), -1), 1 / self.p)
    return dis

  def find_minx(self, clus):
    n = len(clus)
    i_s, j_s, mins = [], [], []
    for i in range(n):
      for j in range(i + 1, n):
        i_s.append(i)
        j_s.append(j)
        if len(clus[i]) > 0 and len(clus[j]) > 0:
          minv = np.min(self.distance(np.array(clus[i])[:, None, :],
                                      np.array(clus[j])))
        else:
          minv = np.inf
        mins.append(minv)
    minidx = np.argmin(mins)
    clus.append((clus[i_s[minidx]] + clus[j_s[minidx]]).copy())
    clus[i_s[minidx]].clear()
    clus[j_s[minidx]].clear()
    self.history.append([i_s[minidx], j_s[minidx]])
    return mins

  def fit(self, X):
    clus = [[x] for x in X]
    while True:
      mins = self.find_minx(clus)
      if np.sum(np.array(mins) == np.inf) == (len(mins) - 1):
        break
    return clus


if __name__ == "__main__":
  from sklearn.datasets import make_moons
  X, y = make_moons(50, noise=8)
  hier = Hierarchical()
  hier.fit(X)
  print('clustering history:\n', np.array(hier.history, np.int32))
