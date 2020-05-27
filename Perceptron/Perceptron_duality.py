import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
from matplotlib.animation import FuncAnimation


class PerceptronDuality(BaseModel):
  def __init__(self, n: int, dim: int = 2):
    super().__init__()
    self.a: np.ndarray = np.random.randn(n)
    self.b: np.ndarray = np.random.randn()
    self.params = [self.a.copy(), self.b]

  def call(self, ys: np.ndarray, i: int):
    a, b = self.params
    return -ys[i] * (np.sum(a * ys * self.G[:, i]) + b)

  def on_train_begin(self, xs):
    self.G = np.dot(xs, xs.T)
    self.history = []

  def on_train_batch_end(self, xs, ys):
    a, b = self.params
    w = np.sum(a[..., None] * ys[..., None] * xs, axis=0)
    self.history.append([w, b])

  def on_train_epoch_begin(self, xs):
    idx = np.arange(len(xs))
    np.random.shuffle(idx)
    return idx

  def fit(self, xs, ys, lr, epochs):
    self.on_train_begin(xs)
    for i in range(epochs):
      for j in self.on_train_epoch_begin(xs):
        if self.call(ys, j) > 0:
          a, b = self.params
          a[j] += lr
          b += lr * ys[j]
          self.params = [a, b]
        self.on_train_batch_end(xs, ys)

  def show_anime(self, xs, ys, save_path: str):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    arr = np.arange(-10, 10, 0.5).reshape((-1, 1))
    ax.scatter(xs[:, 0], xs[:, 1], c=ys)

    w = np.sum(self.a[..., None] * ys[..., None] * xs, axis=0)
    line, = ax.plot(arr, ((w[0] * arr) + self.b) / (-w[1]))

    def update(i):
      w, b = self.history[i]
      line.set_ydata(((w[0] * arr) + b) / (-w[1]))
      return line,

    anim = FuncAnimation(fig, update, frames=range(len(self.history)),
                         interval=70)
    anim.save(save_path, writer='imagemagick', fps=15)
    plt.show()

    plt.scatter(xs[:, 0], xs[:, 1], c=ys)
    w, b = self.history[-1]
    plt.plot(arr, ((w[0] * arr) + b) / (-w[1]))
    plt.show()


if __name__ == "__main__":
  from sklearn import datasets
  import matplotlib.pyplot as plt
  xs, ys = datasets.make_blobs(centers=2)
  xs, ys = xs.astype('float32'), ys.astype('float32')
  ys[ys == 0] = -1

  model = PerceptronDuality(n=len(xs), dim=2)

  model.fit(xs, ys, lr=0.1, epochs=3)
  model.show_anime(xs, ys, save_path='./gif/PerceptronDuality.gif')


def test_case():
  xs = np.array([[3, 3], [4, 3], [1, 1]])
  ys = np.array([1, 1, -1])

  G = np.dot(xs, xs.T)
  a = np.zeros((3))
  b = np.zeros([])
  for i in [0, 2, 2, 2, 0, 2, 2]:
    if ys[i] * (np.sum(a * ys * G[:, i]) + b) <= 0:
      a[i] += 1
      b += ys[i]
      print(a, b)
