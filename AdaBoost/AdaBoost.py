import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel, Dict
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy


class Adaboost(BaseModel):
  def __init__(self, weak_model: str = 'DecisionTreeClassifier', weak_model_args={'max_depth': 1}):
    self.model_dict = {'DecisionTreeClassifier': DecisionTreeClassifier}
    self.weak_model = weak_model
    self.weak_model_args = weak_model_args

  def init_param(self, X, Y):
    assert X.ndim == 2
    self.n, self.m = X.shape
    self.sample_weight = None
    self.clsifys = []
    self.clsifys_w = []
    self.eps = np.finfo(np.float).eps

  def fit(self, X, Y, epoch):
    self.init_param(X, Y)
    for m in range(epoch):
      tmp_clsify = self.model_dict[self.weak_model](**self.weak_model_args)
      if self.sample_weight is None:
        self.sample_weight = np.ones((self.n)) / self.n
      tmp_clsify.fit(X, Y, sample_weight=self.sample_weight)
      P = tmp_clsify.predict(X)
      e = np.clip(self.sample_weight @ (P != Y), self.eps, 1 - self.eps)
      alpha = 0.5 * np.log((1 - e) / e)
      self.sample_weight *= np.exp(-alpha * Y * P)
      self.sample_weight /= np.sum(self.sample_weight)
      self.clsifys.append(deepcopy(tmp_clsify))
      self.clsifys_w.append(alpha)
    self.clsifys_w = np.array(self.clsifys_w)

  def predict(self, test_x, bound: int = -1):
    preds = np.array([clf.predict(test_x) for clf in self.clsifys[:bound]]).T
    preds = np.sum(preds * self.clsifys_w[:bound], axis=-1)
    return np.sign(preds)

  def show_anime(self, X_train, X_test, y_train, y_test,
                 save_path='gif/Adaboost.gif'):
    y_train_color = np.ones_like(y_train, dtype=np.str)
    y_train_color[y_train == 1] = 'b'
    y_train_color[y_train == -1] = 'y'
    y_test_color = np.ones_like(y_test, dtype=np.str)
    y_test_color[y_test == 1] = 'b'
    y_test_color[y_test == -1] = 'y'
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_color)
    line = ax.scatter(X_test[:, 0], X_test[:, 1], c='r')
    ax.set_title('original')

    def update(i):
      ax.set_title(f'use {i+1} model')
      y_pred_color = np.copy(y_test_color)
      y_pred = self.predict(X_test, bound=i + 1)
      y_pred_color[y_pred != y_test] = 'r'
      ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_color)
      return ax

    anim = FuncAnimation(fig, update, frames=len(self.clsifys_w), interval=1000)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score

  X, Y = datasets.make_moons(n_samples=150, noise=0.2, random_state=1)
  X, Y = X.astype('float32'), Y.astype('float32')
  Y[Y == 0.] = -1.
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
  ada = Adaboost()

  ada.fit(X_train, y_train, epoch=10)
  ada.show_anime(X_train, X_test, y_train, y_test)


def test_weight():
  e = np.linspace(0.01, 0.999)
  a = 0.5 * np.log((1 - e) / e)
  plt.plot(e, a)
  plt.title(r"$\alpha_{k} = \frac{1}{2}\ln\frac{1 - e_{k}}{e_{k}}$")
  plt.savefig('/tmp/alpha.png')