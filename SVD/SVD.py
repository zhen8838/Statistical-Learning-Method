import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from functools import reduce


class SVD(BaseModel):
  def __init__(self):
    super().__init__()

  def fit(self, X):
    u, sigma, v = np.linalg.svd(X)
    self.history = [['E', 'v^T', '\Sigma ', 'u'],
                    [np.eye(len(X)), v, np.diag(sigma), u]]

  def show_anime(self, save_path='gif/SVD.gif'):
    fig: plt.Figure = plt.figure()
    fig.set_tight_layout(True)
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=25., azim=120.)

    def update(i):
      prefixs, mats = self.history[0][:i + 1], self.history[1][:i + 1]
      prefixs.reverse()
      fname = ''.join(prefixs)
      fe = reduce(lambda a, b: b @ a, mats)
      ax.cla()
      ax.set_title('$' + fname + '$')
      for line, color in zip(fe, ['g', 'orange', 'r']):
        ax.quiver(0, 0, 0,
                  line[0] - 0, line[1] - 0, line[2] - 0,
                  length=0.1, normalize=False, color=color,
                  arrow_length_ratio=0.1)
      ax.set_xticks([-.1, 0, .1])
      ax.set_yticks([-.1, 0, .1])
      ax.set_zticks([-.1, 0, .1])
    anim = FuncAnimation(fig, update, frames=len(self.history[0]), interval=1000)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  X = np.array([[3, 1, 2],
                [1, 0, 1],
                [2, 3, 3]])
  svd = SVD()
  svd.fit(X)
  svd.show_anime()
