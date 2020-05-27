import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from KNN.KNN import KNN


class KNNKD(KNN):
  def __init__(self, dimension_k: int = 2, distance_p: float = 2):
    super().__init__()
    self.k = dimension_k
    self.p = distance_p

  def build(self, x: np.ndarray, depth: int = 0):
    n = len(x)
    if n == 0:
      return None
    axis = depth % self.k
    sorted_x = sorted(x, key=lambda point: point[axis])
    root = sorted_x[n // 2]
    left = sorted_x[:n // 2]
    right = sorted_x[n // 2 + 1:]
    return dict(root=root, left=self.build(left, depth + 1), right=self.build(right, depth + 1))

  def fit(self, x: np.ndarray):
    self.tree = self.build(x, depth=0)

  def closer_dist(self, point, p1, p2):
    if p1 is None:
      return p2
    if p2 is None:
      return p1

    d1 = self.distance(point, p1)
    d2 = self.distance(point, p2)

    if d1 < d2:
      return p1
    else:
      return p2

  def __find(self, root: dict, point, depth=0):
    if root is None:
      return None

    axis = depth % self.k

    next_branch = None
    opposite_branch = None

    if point[axis] < root['root'][axis]:
      next_branch = root['left']
      opposite_branch = root['right']
    else:
      next_branch = root['right']
      opposite_branch = root['left']

    best = self.closer_dist(point,
                            self.__find(next_branch,
                                        point,
                                        depth + 1),
                            root['root'])

    if self.distance(point, best) > abs(point[axis] - root['root'][axis]):
      best = self.closer_dist(point,
                              self.__find(opposite_branch,
                                          point,
                                          depth + 1),
                              best)
    return best

  def find(self, point):
    return self.__find(self.tree, point, 0)

  def show_anime(self, X_train, X_test, y_train, y_test, y_pred, save_path: str):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    line = ax.scatter(X_test[:, 0], X_test[:, 1], c='b')

    def update(i):
      title, c = [('test', 'r'), ('predict', y_pred), ('original', y_test), ][i]
      ax.set_title(title)
      ax.scatter(X_test[:, 0], X_test[:, 1], c=c)
      return ax

    anim = FuncAnimation(fig, update, frames=3, interval=1000)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  iris = load_iris()
  y = iris.target[:100]
  X = iris.data[:100, :2]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  knnkd = KNNKD()
  knnkd.fit(X_train)

  knnkd.find([3., 4.])
