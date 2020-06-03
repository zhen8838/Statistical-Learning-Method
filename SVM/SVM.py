import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel


class SVM(BaseModel):
  def __init__(self, epsilon=0.001, C=1.0, kernel='linear'):
    super().__init__()
    self.kernel = kernel
    self.eps = epsilon
    self.C = C

  def init_param(self, X, Y):
    assert X.ndim == 2
    self.n = len(X)
    self.dim = X.shape[-1]
    self.a = np.ones(self.n, dtype=np.float32)
    self.b = 0
    self.gram = self.K(X, X)
    self.gx = np.array([self.g(X, Y, i) for i in range(self.n)])
    self.e = self.gx - Y
    self.history = []

  def g(self, X, Y, i):
    return np.sum(self.a * Y * self.gram[i, :]) + self.b

  def E(self, X, Y, i):
    return self.g(X, Y, i) - Y[i]

  def K(self, a, b) -> np.ndarray:
    if self.kernel == 'linear':
      return a @ b.T

  def select_first_alpha(self, X, Y):
    ygx = Y * self.gx
    cond1 = (0 == self.a)
    cond2 = (0 < self.a) & (self.a < self.C)
    cond3 = (self.a == self.C)
    err = np.abs(ygx - 1)
    # KKT
    err[(cond1 & (ygx >= 1)) | (cond2 & (ygx == 1)) | (cond3 & (ygx <= 1))] = 0
    idx = np.argmax(err)
    if err[idx] < self.eps:
      return None
    return idx

  def select_second_alpha(self, X, Y, idx1):
    e1 = self.e[idx1]
    idx2 = np.argmax(np.abs(e1 - self.e))
    return idx2

  def get_bound(self, Y, idx1, idx2):
    if Y[idx1] == Y[idx2]:
      L = max(0., self.a[idx2] + self.a[idx1] - self.C)
      H = min(self.C, self.a[idx2] + self.a[idx1])
    else:
      L = max(0., self.a[idx2] - self.a[idx1])
      H = min(self.C, self.C + self.a[idx2] - self.a[idx1])
    return L, H

  def update_alpha_b(self, X, Y, idx1, idx2):
    y1, y2 = Y[idx1], Y[idx2]
    e1, e2 = self.e[idx1], self.e[idx2]
    gram_11 = self.gram[idx1][idx1]
    gram_22 = self.gram[idx2][idx2]
    gram_12 = self.gram[idx1][idx2]
    gram_21 = self.gram[idx2][idx1]

    L, H = self.get_bound(Y, idx1, idx2)
    eta = gram_11 + gram_22 - 2 * gram_12
    a2_new = self.a[idx2] + (y2 * (e1 - e2)) / eta
    a2_new = np.clip(a2_new, L, H)
    a1_old, a2_old = self.a[idx1], self.a[idx2]
    da2 = a2_new - a2_old
    da1 = - y1 * y2 * da2
    a1_new = a1_old + da1

    b1_new = (-e1 - y1 * gram_11 * da1 - y2 * gram_21 * da2 + self.b)  # 7.115
    b2_new = (-e2 - y1 * gram_12 * da1 - y2 * gram_22 * da2 + self.b)  # 7.116
    if 0 < a1_new < self.C and 0 < a2_new < self.C:
      b_new = b1_new
    else:
      b_new = (b1_new + b2_new) / 2
    # update
    self.a[idx1] = a1_new
    self.a[idx2] = a2_new
    self.b = b_new
    self.gx[idx1] = self.g(X, Y, idx1)
    self.gx[idx2] = self.g(X, Y, idx2)
    self.e[idx1] = self.gx[idx1] - y1
    self.e[idx2] = self.gx[idx2] - y2

  def fit(self, X, Y, epoch):
    self.init_param(X, Y)
    for _ in range(epoch):
      idx1 = self.select_first_alpha(X, Y)
      if idx1:
        idx2 = self.select_second_alpha(X, Y, idx1)
        self.update_alpha_b(X, Y, idx1, idx2)
      else:
        break
      self.history.append(self.get_w_b(X, Y))
    print('End in iter: ', _)

  def get_w_b(self, X, Y):
    yx = X * Y[:, None]
    self.w = np.dot(yx.T, self.a[:, None]).ravel()
    return self.w, self.b

  def show_anime(self, X, Y, save_path='gif/SVM.gif'):
    w, b = np.dot((X * Y[:, None]).T, np.ones((self.n, 1))).ravel(), 0
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    arr = np.linspace(min(X[:, 0]) - 2, max(X[:, 0] + 2))
    ax.scatter(X[:, 0], X[:, 1], c=Y)
    bound = (w[0] * arr + b) / (-w[1])
    line, = ax.plot(arr, bound, color='r')

    def update(i):
      ax.set_title(f'iter = {i}')
      w, b = self.history[i]
      bound = (w[0] * arr + b) / (-w[1])
      line.set_ydata(bound)

    anim = FuncAnimation(fig, update,
                         frames=range(len(self.history)), interval=70)
    anim.save(save_path, writer='imagemagick', fps=15)
    plt.show()


if __name__ == "__main__":
  from sklearn import datasets
  import matplotlib.pyplot as plt
  X, Y = datasets.make_blobs(centers=2)
  X, Y = X.astype('float32'), Y.astype('float32')
  Y[Y == 0.] = -1.
  svm = SVM()
  svm.fit(X, Y, epoch=100)
  svm.show_anime(X, Y)
