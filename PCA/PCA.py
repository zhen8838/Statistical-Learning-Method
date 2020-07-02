import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import normalize


class PCA(BaseModel):
  def __init__(self, num_components):
    super().__init__()
    self.num_components = num_components
    self.history = []

  def fit(self, X, Y):
    self.Y = Y
    n, m = X.shape
    C = np.cov(X.T) / n  # Get covariance matrix
    v, w = np.linalg.eig(C)  # Eigen decomposition
    # Project X onto PC space
    X_pca = np.dot(X, w[:, :self.num_components])
    self.history.append(X)
    self.history.append(X_pca)
    return X_pca

  def show_anime(self, save_path='gif/PCA.gif'):
    fig: plt.Figure = plt.figure()
    fig.set_tight_layout(True)
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=25., azim=120.)

    def update(i):
      X = self.history[i]
      plt.cla()
      if i == 0:
        ax.set_title('before PCA')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=self.Y)
      elif i == 1:
        ax.set_title('after PCA')
        ax.scatter(X[:, 0], X[:, 1], c=self.Y)

    anim = FuncAnimation(fig, update, frames=len(self.history), interval=1000)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  iris = load_iris()
  Y = iris.target
  X = iris.data[:, :3]
  X = normalize(X, axis=0)
  pca = PCA(2)
  x_pca = pca.fit(X, Y)
  pca.show_anime()
