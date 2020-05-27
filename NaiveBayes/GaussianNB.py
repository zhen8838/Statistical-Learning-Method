import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel


class GaussianNB(BaseModel):
  def __init__(self):
    super().__init__()

  def fit(self, X: np.ndarray, Y: np.ndarray):
    assert X.dtype == np.float32, f'now is {X.dtype}'
    assert X.ndim == 2
    assert Y.ndim == 1
    self.nfeatrues = X.shape[-1]
    # 获得标签与特征的映射字典
    self.labels_map = self.makemap(Y)
    self.featrues_map = [self.makeGaussianmap(X[:, i]) for i in range(self.nfeatrues)]

    # 计算先验概率
    self.prob_y = [np.sum(Y == k) / len(Y) for k, v in self.labels_map.items()]
    # 计算条件概率
    self.prob_x_y = []
    # NOTE 条件概率的索引为: [特征编号,标签编号,特征类别编号]
    for fidx, fmap in enumerate(self.featrues_map):
      tmp = []
      for lbidx, (lk, lv) in enumerate(self.labels_map.items()):
        give = (Y == lk)
        # 统计均值与方差
        tmp.append(np.array([np.mean(X[give, fidx]), np.var(X[give, fidx])]))
      self.prob_x_y.append(tmp)

  def _pdf(self, x: np.ndarray, fidx: int, lbidx: int):
    assert x.ndim == 1
    mean = self.prob_x_y[fidx][lbidx][0]
    var = self.prob_x_y[fidx][lbidx][1]

    numerator = np.exp(-np.square(x - mean) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var + 1e-6)
    posterior = numerator / denominator

    return posterior

  def predict(self, test_x: np.ndarray):
    assert test_x.ndim == 2
    # 计算后验概率,并选择最大的后验概率作为预测结果
    label_key = np.array(list(self.labels_map.keys()))
    probs = []
    for lbk, lbidx in self.labels_map.items():
      prob = self.prob_y[lbidx]  # prior
      for fidx, fmap in enumerate(self.featrues_map):
        prob *= self._pdf(test_x[:, fidx], fidx, lbidx)

      probs.append(prob)

    probs = np.stack(probs, -1)
    max_prob_idx = np.argmax(probs, -1)
    return label_key[max_prob_idx]

  @staticmethod
  def makemap(arr):
    tag = np.unique(arr)
    tag_map = dict(zip(tag, range(len(tag))))
    return tag_map

  @staticmethod
  def makeGaussianmap(arr):
    tag_map = {'mean': 0, 'std': 1}
    return tag_map


if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  csv = np.loadtxt('NaiveBayes/credit-g.csv', dtype=np.str, delimiter=',', skiprows=1)
  X = csv[:, :20]
  Y = csv[:, -1]

  whether_continuous = np.zeros((X.shape[1]), np.bool)
  continuous_lst = [1, 4, 12]
  whether_continuous[continuous_lst] = True
  X = X[:, whether_continuous]  # NOTE now X is [n,featrue] Gaussian
  X = np.asfarray(X, dtype='float32')
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

  nb = GaussianNB()
  nb.fit(X_train, y_train)
  y_pred = nb.predict(X_test)
  print('Predict acc :', accuracy_score(y_test, y_pred))
