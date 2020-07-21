import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple


class PLSA(BaseModel):
  def __init__(self, k, max_iter, tol=1e-3):
    super().__init__()
    self.nz = k
    self.max_iter = max_iter
    self.tol = tol
    self.history = []

  def fit(self, X):
    nw, nd = X.shape

    # P(w|z)
    self.p_w_z = np.random.rand(nw, self.nz)
    # P(z|d)
    self.p_z_d = np.random.rand(self.nz, nd)

    for _ in range(self.max_iter):
      self.history.append((self.p_w_z, self.p_z_d))
      # E step
      t = np.einsum('wz,zd->dzw', self.p_w_z, self.p_z_d)
      p_d_zw = t / np.sum(t, axis=1, keepdims=True)
      # M step
      new_p_w_z = (np.einsum('wd,dzw->wz', X, p_d_zw) / np.einsum('wd,dzw->z', X, p_d_zw))
      new_p_z_d = (np.einsum('wd,dzw->zd', X, p_d_zw) / np.sum(X, 0))
      if np.allclose(new_p_w_z, self.p_w_z, atol=self.tol):
        break
      self.p_w_z = new_p_w_z
      self.p_z_d = new_p_z_d
    self.history.append((self.p_w_z, self.p_z_d))
    return self.p_w_z, self.p_z_d

  def show_anime(self, save_path='gif/PLSA.gif'):
    fig: plt.Figure = plt.figure()
    fig.set_tight_layout(True)
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    z = 1 - x - y
    z[z < 0] = np.nan
    ax.view_init(elev=25., azim=10.)

    def update(i):
      p_w_z, p_z_d = self.history[i]
      plt.cla()
      ax.set_title(f'iter {i}')
      ax.plot_surface(x, y, z, alpha=0.3)
      ax.set_zlim(0, 1)
      ax.scatter3D(p_z_d[0], p_z_d[1], p_z_d[2], label=r'$P(z_k|d_j)$')
      ax.scatter3D(p_w_z[:, 0], p_w_z[:, 1], p_w_z[:, 2], label=r'$P(w_i|z_k)$')
      ax.legend()

    anim = FuncAnimation(fig, update, frames=len(self.history))
    anim.save(save_path, writer='imagemagick', fps=6)
    plt.show()


if __name__ == "__main__":
  X = np.asarray([[0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 2, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0]])
  plsa = PLSA(k=3, max_iter=100)
  p_w_z, p_z_d = plsa.fit(X)
  plsa.show_anime()
