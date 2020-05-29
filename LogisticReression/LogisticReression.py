import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel


class LogisticReression(BaseModel):
  def __init__(self, dim: int = 2):
    super().__init__()
    self.dim = dim
    self.θ = np.random.normal(size=(dim + 1, 1))

  @staticmethod
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def fit(self, X: np.ndarray, Y: np.ndarray, epoch, lr):
    assert X.ndim == 2
    assert Y.ndim == 2
    self.history = []
    n = X.shape[0]
    # NOTE X => [1,X] for b
    X = np.hstack([np.ones((n, 1)), X])
    for e in range(epoch):
      # x:[n,dim] * θ:[dim,1] => pred:[n,1]
      pred = self.sigmoid(X@self.θ)
      grad = (X.T @ (pred - Y)) / self.dim
      self.θ += lr * grad
      self.history.append(self.θ.copy())

  def show_anime(self, xs, save_path='gif/LogisticReression.gif'):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    theta = self.history[0]

    arr = np.linspace(min(xs[:, 0]) - 2, max(xs[:, 0] + 2))

    ax.scatter(xs[:, 0], xs[:, 1], c=ys)
    y = (theta[1, 0] * arr + theta[0, 0]) / (-theta[2, 0])
    line, = ax.plot(arr, y, color='r')

    def update(i):
      ax.set_title(f'iter = {i}')
      theta = self.history[i]
      y = (theta[1, 0] * arr + theta[0, 0]) / (-theta[2, 0])
      line.set_ydata(y)

    anim = FuncAnimation(fig, update,
                         frames=range(len(self.history)), interval=70)
    anim.save(save_path, writer='imagemagick', fps=15)
    plt.show()


if __name__ == "__main__":
  from sklearn import datasets
  import matplotlib.pyplot as plt
  xs, ys = datasets.make_blobs(centers=2)
  xs, ys = xs.astype('float32'), ys.astype('float32')
  ys = ys[:, None]
  model = LogisticReression(dim=2)

  model.fit(xs, ys, lr=0.01, epoch=100)
  model.show_anime(xs)
