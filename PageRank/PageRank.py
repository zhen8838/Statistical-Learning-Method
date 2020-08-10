import os
import sys

from matplotlib.pyplot import text
from numpy.lib.function_base import select
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class PageRank(BaseModel):
  def __init__(self, M, d) -> None:
    self.M = M
    self.d = d

  def init_param(self):
    self.w = self.d * self.M
    self.b = (1 - self.d) / len(M) * np.ones(len(M))
    self.history = []
    self.R = np.ones(len(M)) / len(M)

  def fit(self, tol=0.001):
    self.init_param()
    while True:
      self.history.append(self.R.copy())
      R_ = self.w @ self.R + self.b
      if np.allclose(R_, self.R, atol=tol):
        break
      else:
        self.R = R_

  def show_anime(self, save_path: str = 'gif/PageRank.gif'):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    arr = self.history[0]
    im = ax.pcolor(arr.reshape((2, 2)), cmap='RdBu_r',
                   vmin=0, vmax=1,
                   edgecolors='w', linewidths=30)

    ax.set_xticks([])
    ax.set_yticks([])
    texts = []
    strs = ['A', 'B', 'C', 'D']
    for i in range(2):
      for j in range(2):
        texts.append(ax.text(j + 0.5, i + 0.5, f'{strs[i+j]}:{arr[i+j]:.2f}',
                             ha="center", va="center", color="k", fontsize=14))

    def update(idx):
      ax.set_title(f'iter {idx}', fontsize=14)
      arr = self.history[idx]
      im.set_array(arr)
      [t.set_text(f'{s}:{a:.2f}') for s, t, a in zip(strs, texts, arr)]

    anim = FuncAnimation(fig, update, frames=len(self.history), interval=300)
    anim.save(save_path, writer='imagemagick', fps=5)
    plt.show()


if __name__ == "__main__":
  M = np.array([[0, 1 / 2, 0, 0],
                [1 / 3, 0, 0, 1 / 2],
                [1 / 3, 0, 1, 1 / 2],
                [1 / 3, 1 / 2, 0, 0]])
  d = 4 / 5

  pr = PageRank(M, d)
  pr.fit()
  pr.show_anime()
