import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel, Dict


class DecisionTree(BaseModel):
  def fit(self, X, Y):
    """ NOTE 为了可视化条件概率分布P(Y|X)才用这么麻烦的方法 """
    assert X.ndim == 2
    assert Y.ndim == 1
    self.nfeatrues = X.shape[-1]
    self.labels_map = self.makemap(Y)
    self.featrues_map = [self.makemap(X[:, i]) for i in range(self.nfeatrues)]

    # 计算先验概率 P(D)
    self.prob_y = [np.sum(Y == k) / len(Y) for k, v in self.labels_map.items()]
    # 计算经验熵 H(D)
    self.entropy_y = self.entropy(self.prob_y)
    self.prob_x = []
    # 计算条件概率 P(D|A)
    self.prob_y_x = []
    # NOTE 条件概率的索引为: [特征编号,特征类别编号,标签编号]
    for fidx, fmap in enumerate(self.featrues_map):
      # 计算特征经验概率
      binarize_x = fmap[X[:, fidx]]
      self.prob_x.append(np.bincount(binarize_x, minlength=len(fmap)) / len(Y))

      y_x_prob_list = []
      for fcidx, (fk, fv) in enumerate(fmap.items()):
        # fcidx 是这个特征的中类别的索引
        give_x = (X[:, fidx] == fk)
        # 计算给定X时Y的概率分布
        binarize_y_x = self.labels_map[Y[give_x]]
        # 利用bincount统计次数并计算概率
        y_x_prob_list.append(np.bincount(
            binarize_y_x, minlength=len(self.labels_map)) / np.sum(give_x))

      self.prob_y_x.append(np.array(y_x_prob_list))

    fidx, fmap = 0, self.featrues_map[0]
    self.entropy_y_x = []
    # 经验条件熵 H(D|A)
    for fidx, fmap in enumerate(self.featrues_map):
      self.entropy_y_x.append(self.prob_x[fidx] *
                              [self.entropy(self.prob_y_x[fidx][i]) for i in range(len(fmap))])
    self.info_gain = []  # 信息增益 = H(D) - H(D|A)
    for fidx, fmap in enumerate(self.featrues_map):
      self.info_gain.append(self.entropy_y - np.sum(self.entropy_y_x[fidx]))

  @staticmethod
  def makemap(arr):
    tag = np.unique(arr)
    tag_map = Dict(zip(tag, range(len(tag))))
    return tag_map

  @staticmethod
  def entropy(p):
    assert len(p) > 1
    return -np.sum(p * np.log2(p + np.finfo(np.float32).eps))

  def show_anime(self, save_path: str = 'gif/DecisionTree.gif'):
    labels_map = self.labels_map
    label_key = np.array(list(labels_map.keys()))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    colors = plt.cm.Paired([i / len(label_key) for i in range(len(label_key))])
    colors = {cat: color for cat, color in zip(labels_map.values(), colors)}

    def update(fidx):
      ax.cla()
      fmap = self.featrues_map[fidx]
      for lbk, lbidx in labels_map.items():
        sj = len(fmap)
        tmp_x = np.arange(1, sj + 1)
        ax.bar(tmp_x - 0.35 * lbidx,
               self.prob_y_x[fidx][:, lbidx], width=0.35,
               facecolor=colors[lbidx], edgecolor="white",
               label=f"class: {lbk}")
      ax.set_title(f'$P(Y|X_j)$  $j$ = {fidx} ; $S_j$ = {len(fmap)}')
      ax.set_xticks([i for i in range(sj + 2)])
      ax.set_xticklabels([""] + list(fmap.keys()) + [""], rotation=-30)
      ax.set_ylim(0, 1.0)
      ax.legend()

    anim = FuncAnimation(fig, update, frames=len(self.featrues_map), interval=1000)
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
  X = X[:, whether_continuous]  # NOTE now X is [n,featrue] Multinomial
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  dt = DecisionTree()
  dt.fit(X_train, y_train)
  best_fidx = np.argmax(dt.info_gain)

  print('Best featrue index :', best_fidx, '\nIt have :', dt.featrues_map[best_fidx])
  dt.show_anime()


def test():
  data = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 2, 1],
                   [1, 0, 1, 2, 0],
                   [1, 0, 0, 1, 0],
                   [1, 1, 1, 1, 1],
                   [2, 1, 1, 1, 1],
                   [2, 1, 1, 2, 1],
                   [2, 0, 0, 2, 0],
                   [2, 1, 0, 3, 0],
                   [2, 1, 0, 3, 0],
                   [3, 1, 0, 3, 0],
                   [3, 1, 0, 2, 0],
                   [3, 0, 1, 2, 0],
                   [3, 0, 1, 3, 0],
                   [3, 1, 1, 1, 1]])
  X, Y = data[:, :-1], data[:, -1]
  dt = DecisionTree()
  dt.fit(X, Y)
  print('Best featrue index :', np.argmax(dt.info_gain))
