import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
from matplotlib.animation import FuncAnimation


class Perceptron(BaseModel):
  def __init__(self, dim: int = 2):
    super().__init__()
    self.w = np.random.randn(dim)
    self.b = np.random.randn()
    self.params = [self.w, self.b]

  def call(self, x: np.ndarray, y: np.ndarray):
    w, b = self.params
    return -y * (np.dot(x, w) + b)

  def on_train_begin(self):
    self.history = []

  def on_train_batch_end(self):
    self.history.append(self.params.copy())

  def on_train_epoch_begin(self, xs, ys):
    idx = np.arange(len(xs))
    np.random.shuffle(idx)
    return xs[idx], ys[idx]

  def fit(self, xs, ys, lr, epochs):
    self.on_train_begin()
    for i in range(epochs):
      for x, y in zip(*self.on_train_epoch_begin(xs, ys)):
        if self.call(x, y) > 0:
          self.params = [param + lr * grad for (param, grad) in zip(self.params, [y * x, y])]
        self.on_train_batch_end()

  def show_anime(self, save_path: str):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    arr = np.arange(-10, 10, 0.5).reshape((-1, 1))
    ax.scatter(xs[:, 0], xs[:, 1], c=ys)
    line, = ax.plot(arr, ((self.w[0] * arr) + self.b) / (-self.w[1]))

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

  model = Perceptron(dim=2)

  model.fit(xs, ys, lr=0.01, epochs=3)
  model.show_anime(save_path='./gif/Perceptron.gif')
