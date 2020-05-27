import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class KNN(BaseModel):
  def __init__(self, neighbors_k: int = 1, distance_p: float = 2):
    super().__init__()
    self.p = distance_p
    self.k = neighbors_k

  def distance(self, x: np.ndarray, y: np.ndarray):
    dis = np.power(np.sum(np.power(np.abs((x - y)), self.p), -1), 1 / self.p)
    return dis

  def fit(self, x, y):
    self.x = x
    self.y = y

  def predict(self, x):
    # add x axis for mat calc distance
    distmat = self.distance(self.x, x[:, None, :])
    sort_idx = np.argsort(distmat, -1)
    top_k_idx = sort_idx[:, :knn.k]

    tok_k_dis = np.array([dist_v[dix] for dist_v, dix in zip(distmat, top_k_idx)])
    tok_k_point = np.array([self.x[dix] for dix in top_k_idx])
    tok_k_y = np.array([self.y[dix] for dix in top_k_idx])

    y_pred = np.array([
        # 2. max neighbor label is predict label
        label[np.argmax(cnt)] for label, cnt in
        # 1. use unique count neighbor label
        [np.unique(k_y, return_counts=True) for k_y in tok_k_y]])
    return y_pred, tok_k_point, tok_k_y

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

  knn = KNN(5, 2)
  knn.fit(X_train, y_train)
  y_pred, tok_k_point, tok_k_y = knn.predict(X_test)
  knn.show_anime(X_train, X_test, y_train, y_test, y_pred, 'gif/KNN.gif')
