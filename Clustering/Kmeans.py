import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Kmeans(BaseModel):
  def __init__(self, k: int, p: float = 2):
    super().__init__()
    self.k = k
    self.p = p
    self.history = []

  def distance(self, x: np.ndarray, y: np.ndarray):
    dis = np.power(np.sum(np.power(np.abs((x - y)), self.p), -1), 1 / self.p)
    return dis

  def fit(self, X, epoch, tol: float = 0.001):
    center = X[np.random.choice(range(len(X)), self.k)]
    for _ in range(epoch):
      dis = self.distance(center[:, None, :], X)  # [k,n]
      self.clus = np.argmin(dis, 0)
      new_center = np.array([np.mean(X[self.clus == _], 0) for _ in range(self.k)])
      if np.allclose(center, new_center, atol=tol):
        break
      else:
        center = new_center

      self.history.append(center.copy())

  def show_anime(self, X, save_path='gif/Kmeans.gif'):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    ax.scatter(X[:, 0], X[:, 1], c=self.clus)

    def update(i):
      ax.set_title(f'kmeas iter: {i+1}')
      ax.plot(self.history[i][:, 0], self.history[i][:, 1], 'bx')
      ax.plot(self.history[i + 1][:, 0], self.history[i + 1][:, 1], 'rx')
      # Plot the history of the centroids with lines
      for j in range(self.k):
        ax.plot(np.r_[self.history[i + 1][j, 0], self.history[i][j, 0]],
                np.r_[self.history[i + 1][j, 1], self.history[i][j, 1]],
                'k--')

    anim = FuncAnimation(fig, update, frames=len(self.history) - 1, interval=500)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  from sklearn.datasets import make_moons
  X, y = make_moons(200, noise=8)
  kmeans = Kmeans(k=3)
  kmeans.fit(X, 20)
  kmeans.show_anime(X)
