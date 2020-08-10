import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel, Dict
from scipy.stats import multivariate_normal


class GMM(BaseModel):

  @staticmethod
  def normalize(xs, axis=None):
    """Return normalized marirx so that sum of row or column (default) entries = 1."""
    if axis is None:
      return xs / xs.sum()
    elif axis == 0:
      return xs / xs.sum(0)
    else:
      return xs / xs.sum(1)[:, None]

  @staticmethod
  def mix_mvn_pdf(xs, pis, mus, sigmas):
    return np.array([pi * multivariate_normal(mu, sigma).pdf(xs) for (pi, mu, sigma) in zip(pis, mus, sigmas)])

  def init_param(self, X, K):
    self.n, self.p = X.shape
    self.pis = self.normalize(np.random.random(K))  # pi 要经过归一化
    self.mus = np.random.random((K, K))
    self.sigmas = np.array([np.eye(K)] * K)
    self.history = []
    self.history.append([self.pis.copy(), self.mus.copy(), self.sigmas.copy()])

  def fit(self, X, K, tol=0.01, max_iter=100):
    self.init_param(X, K)
    ll_old = 0
    for i in range(max_iter):
      exp_A = []
      exp_B = []
      ll_new = 0

      # E-step
      ws = np.zeros((K, self.n))
      for j in range(K):
        for i in range(n):
          ws[j, i] = self.pis[j] * multivariate_normal(self.mus[j], self.sigmas[j]).pdf(X[i])
      ws /= ws.sum(0)  # 根据概率密度求权值

      # M-step
      self.pis = (self.pis + np.sum(ws, -1)) / self.n
      self.mus = (self.mus + (ws @ X)) / np.sum(ws, 1)
      err = X - self.mus[:, None, :]
      self.sigmas = (self.sigmas + np.sum(
          ws[..., None, None] * (err[..., None] @ err[..., None, :]), axis=1)) / np.sum(ws, 1)

      ll_new = 0.0
      for i in range(n):
        for j in range(K):
          s = self.pis[j] * multivariate_normal(self.mus[j], self.sigmas[j]).pdf(X[i])
        ll_new += np.log(s)

      if np.abs(ll_new - ll_old) < tol:
        break
      ll_old = ll_new
      self.history.append([self.pis.copy(), self.mus.copy(), self.sigmas.copy()])
    return ll_new

  def show_anime(self, X, save_path='gif/ExpectationMaximization.gif'):
    intervals = 50
    ys = np.linspace(-8, 8, intervals)
    xx, yy = np.meshgrid(ys, ys)
    _ys = np.vstack([xx.ravel(), yy.ravel()]).T

    z = np.zeros(len(_ys))
    for pi, mu, sigma in zip(*self.history[0]):
      z += pi * multivariate_normal(mu, sigma).pdf(_ys)
    z = z.reshape((intervals, intervals))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    ax.scatter(X[:, 0], X[:, 1], alpha=0.2)
    ax.contour(xx, yy, z)
    ax.axis([-8, 6, -6, 8])
    ax.axes.set_aspect('equal')

    def update(idx):
      ax.cla()
      ax.set_title(f'iter {idx+1}')
      z = np.zeros(len(_ys))
      for pi, mu, sigma in zip(*self.history[idx]):
        z += pi * multivariate_normal(mu, sigma).pdf(_ys)
      z = z.reshape((intervals, intervals))
      ax.scatter(X[:, 0], X[:, 1], alpha=0.2)
      ax.contour(xx, yy, z)

    anim = FuncAnimation(fig, update, frames=len(self.history), interval=300)
    anim.save(save_path, writer='imagemagick', fps=5)
    plt.show()


if __name__ == "__main__":
  np.random.seed(101)
  n = 1000
  _mus = np.array([[0, 4], [-2, 0]])
  _sigmas = np.array([[[3, 0], [0, 0.5]], [[1, 0], [0, 2]]])
  _pis = np.array([0.6, 0.4])
  np.set_numeric_ops
  X = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi * n))
                      for pi, mu, sigma in zip(_pis, _mus, _sigmas)])

  gmm = GMM()
  gmm.fit(X, 2, max_iter=30)
  gmm.show_anime(X)
