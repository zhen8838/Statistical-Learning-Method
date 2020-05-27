import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel
from NaiveBayes.GaussianNB import GaussianNB
from NaiveBayes.MultinomialNB import MultinomialNB


class MergedNB(BaseModel):
  def __init__(self, whether_continuous: np.ndarray):
    # NOTE 使用whether_continuous分离连续变量与离散变量
    super().__init__()
    self.gasnb = GaussianNB()
    self.mulnb = MultinomialNB()
    self.whether_continuous = whether_continuous

  def fit(self, X: np.ndarray, Y: np.ndarray):
    assert X.ndim == 2
    # NOTE now X is [n,featrue] Multinomial
    discrete_X = X[:, self.whether_continuous]

    # NOTE now X is [n,featrue] Gaussian
    continuous_X = X[:, ~self.whether_continuous]
    if continuous_X.dtype is not np.float32:
      continuous_X = np.asfarray(continuous_X, dtype='float32')

    self.gasnb.fit(continuous_X, Y)
    self.mulnb.fit(discrete_X, Y)

  def predict(self, test_x):
    assert test_x.ndim == 2
    labels_map = self.gasnb.labels_map
    label_key = np.array(list(labels_map.keys()))

    discrete_X = test_x[:, self.whether_continuous]
    continuous_X = test_x[:, ~self.whether_continuous]
    if continuous_X.dtype is not np.float32:
      continuous_X = np.asfarray(continuous_X, dtype='float32')

    probs = []
    # multinomial prob
    for lbk, lbidx in labels_map.items():
      prob = self.mulnb.prob_y[lbidx]
      for fidx, fmap in enumerate(self.mulnb.featrues_map):
        # 全部向量映射
        giveidx = np.array([fmap[_] for _ in discrete_X[:, fidx]])
        prob *= self.mulnb.prob_x_y[fidx][lbidx][giveidx]

      probs.append(prob)

    # gaussian prob
    for lbk, lbidx in labels_map.items():
      for fidx, fmap in enumerate(self.gasnb.featrues_map):
        probs[lbidx] *= self.gasnb._pdf(continuous_X[:, fidx], fidx, lbidx)

    probs = np.stack(probs, -1)
    max_prob_idx = np.argmax(probs, -1)
    return label_key[max_prob_idx]

  def show_anime(self, save_path: str = 'gif/NaiveBayes.gif'):
    labels_map = self.gasnb.labels_map
    label_key = np.array(list(labels_map.keys()))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    colors = plt.cm.Paired([i / len(label_key) for i in range(len(label_key))])
    colors = {cat: color for cat, color in zip(labels_map.values(), colors)}

    def update(fidx):
      ax.cla()
      fmap = self.mulnb.featrues_map[fidx]
      for lbk, lbidx in labels_map.items():
        sj = len(fmap)
        tmp_x = np.arange(1, sj + 1)
        ax.bar(tmp_x - 0.35 * lbidx,
               self.mulnb.prob_x_y[fidx][lbidx][:], width=0.35,
               facecolor=colors[lbidx], edgecolor="white",
               label=f"class: {lbk}")
      ax.set_title(f'$j = {fidx} ; $S_j = {len(fmap)}')
      ax.set_xticks([i for i in range(sj + 2)])
      ax.set_xticklabels([""] + list(fmap.keys()) + [""], rotation=-30)
      ax.set_ylim(0, 1.0)
      ax.legend()

    anim = FuncAnimation(fig, update, frames=len(self.mulnb.featrues_map), interval=1000)
    anim.save(save_path, writer='imagemagick', fps=2)
    plt.show()


if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  csv = np.loadtxt('NaiveBayes/credit-g.csv', dtype=np.str, delimiter=',', skiprows=1)
  X = csv[:, :20]
  Y = csv[:, -1]

  whether_continuous = np.ones((X.shape[1]), np.bool)
  continuous_lst = [1, 4, 12]
  whether_continuous[continuous_lst] = False
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

  nb = MergedNB(whether_continuous)
  nb.fit(X_train, y_train)
  Y_pred = nb.predict(X_test)

  print('Predict acc :', accuracy_score(y_test, Y_pred))
  nb.show_anime()