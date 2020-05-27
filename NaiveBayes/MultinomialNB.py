import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel


class MultinomialNB(BaseModel):
  def __init__(self):
    super().__init__()

  def fit(self, X, Y):
    assert X.ndim == 2
    assert Y.ndim == 1
    self.nfeatrues = X.shape[-1]
    # 获得标签与特征的映射字典
    self.labels_map = self.makemap(Y)
    self.featrues_map = [self.makemap(X[:, i]) for i in range(self.nfeatrues)]

    # 计算先验概率
    self.prob_y = [np.sum(Y == k) / len(Y) for k, v in self.labels_map.items()]
    # 计算条件概率
    self.prob_x_y = []
    # NOTE 条件概率的索引为: [特征编号,标签编号,特征类别编号]
    for fidx, fmap in enumerate(self.featrues_map):
      tmp = []
      for lbidx, (lk, lv) in enumerate(self.labels_map.items()):
        give = (Y == lk)
        binarize_x = np.array([fmap[t] for t in X[give, fidx]])
        # 利用bincount统计次数并计算概率
        tmp.append(np.bincount(binarize_x, minlength=len(fmap)) / np.sum(give))
      self.prob_x_y.append(tmp)

  def predict_one(self, test_x: np.ndarray):
    # 计算后验概率,并选择最大的后验概率作为预测结果
    assert test_x.ndim == 1
    probs = []
    for lbk, lbidx in self.labels_map.items():
      prob = self.prob_y[lbidx]
      for fidx, fmap in enumerate(self.featrues_map):
        giveidx = fmap[test_x[fidx]]
        prob *= self.prob_x_y[fidx][lbidx][giveidx]

      probs.append(prob)

    return list(labels_map.keys())[np.argmax(probs)]

  def predict(self, test_x: np.ndarray):
    assert test_x.ndim == 2
    # 计算后验概率,并选择最大的后验概率作为预测结果
    label_key = np.array(list(self.labels_map.keys()))
    probs = []
    for lbk, lbidx in self.labels_map.items():
      prob = self.prob_y[lbidx]
      for fidx, fmap in enumerate(self.featrues_map):
        # 全部向量映射
        giveidx = np.array([fmap[_] for _ in test_x[:, fidx]])
        prob *= self.prob_x_y[fidx][lbidx][giveidx]

      probs.append(prob)

    probs = np.stack(probs, -1)
    max_prob_idx = np.argmax(probs, -1)
    return label_key[max_prob_idx]

  @staticmethod
  def makemap(arr):
    tag = np.unique(arr)
    tag_map = dict(zip(tag, range(len(tag))))
    return tag_map


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

  nb = MultinomialNB()
  nb.fit(X_train, y_train)
  Y_pred = nb.predict(X_test)

  print('Predict acc :', accuracy_score(y_test, Y_pred))


def test():
  X = np.array([['1', 'S'],
                ['1', 'M'],
                ['1', 'M'],
                ['1', 'S'],
                ['1', 'S'],
                ['2', 'S'],
                ['2', 'M'],
                ['2', 'M'],
                ['2', 'L'],
                ['2', 'L'],
                ['3', 'L'],
                ['3', 'M'],
                ['3', 'M'],
                ['3', 'L'],
                ['3', 'L'], ])
  Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
  nb = MultinomialNB()
  nb.fit(X, Y)
  print('predict y : ', nb.predict_one(np.array(['2', 'S'])))
