import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.animation import FuncAnimation
from scipy.stats import norm


def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
  print()


class LDA(BaseModel):
  def __init__(self, K, niters=2000):
    super().__init__()
    self.K = K
    self.niters = niters
    self.history = []

  def init_params(self, X: np.ndarray, α: float = 0.1, β: float = 0.01):
    #! 1 设置计数矩阵与初始主题矩阵
    self.M, self.V = X.shape
    self.n_mk = np.zeros((self.M, self.K))
    self.n_kv = np.zeros((self.K, self.V))
    self.z_mn = []
    self.α = np.ones(self.K) * α
    self.β = np.ones(self.V) * β

    #! 2 初始化主题分布
    for m in range(self.M):
      _, vs = np.nonzero(X[m])
      z_m = []
      for v in vs:
        # 等概率抽样话题
        k = np.argmax(np.random.multinomial(1, np.ones(self.K) /
                                            self.K, size=X[m, v]), axis=-1)
        self.n_mk[m, k] += 1
        self.n_kv[k, v] += 1
        z_m.append(k)
      self.z_mn.append(z_m)

  def sample_topics(self, X):
    for m in range(self.M):
      _, vs = np.nonzero(X[m])
      for n, v in enumerate(vs):
        # ? 减少计数
        k = self.z_mn[m][n]
        self.n_mk[m, k] -= 1
        self.n_kv[k, v] -= 1
        # ? 按照满概率分布进行抽样
        p = (((self.n_kv[:, v] + self.β[v]) /
              (np.sum(self.n_kv + self.β, -1))) *
             ((self.n_mk[m, :] + self.α) /
              np.sum(self.n_mk[m, :] + self.α, -1)))
        # NOTE 因为直接使用词频特征进行计算,当一篇文档出现相同词时,多次采样
        k_ = np.argmax(np.random.multinomial(1, p / np.sum(p), size=X[m, v]), axis=-1)
        self.z_mn[m][n] = k_
        # ? 增加计数
        self.n_mk[m, k_] += 1
        self.n_kv[k_, v] += 1
    self.get_param()
    self.history.append([self.φ.copy(), self.θ.copy()])

  def get_param(self):
    self.φ = (self.n_kv + self.β)
    self.φ /= np.sum(self.φ, -1, keepdims=True)
    self.θ = (self.n_mk + self.α)
    self.θ /= np.sum(self.θ, -1, keepdims=True)

  def fit(self, X: np.ndarray, α: float = 0.1, β: float = 0.01):
    self.init_params(X, α, β)
    for i in range(self.niters):
      self.sample_topics(X)
    self.get_param()
    self.history = np.array(self.history)

  def show_anime(self, save_path='gif/LDA.gif'):
    fig: plt.Figure = plt.figure()
    fig.set_tight_layout(True)
    ax1: plt.Axes = fig.add_subplot(111)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    im: QuadMesh = ax1.imshow(self.history[0][0])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    text = ax1.set_title(f'LDA Gibbs Sampling : {0}')

    def animate(i):
      text.set_text(f'LDA Gibbs Sampling : {i}')
      im.set_array(self.history[i][0])
      return [text, im]
      
    anim = FuncAnimation(fig, animate, frames=len(self.history),
                         interval=50)
    anim.save(save_path, writer='imagemagick', fps=30)
    plt.show()


if __name__ == "__main__":
  from sklearn.datasets import fetch_20newsgroups
  from sklearn.feature_extraction.text import CountVectorizer
  from scipy.stats import multinomial

  n_data = 200
  n_features = 25
  n_components = 10
  n_top_words = 20

  data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                               remove=('headers', 'footers', 'quotes'),
                               return_X_y=True)
  data = data[:n_data]
  tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                  # 特征最大为1000
                                  max_features=n_features,
                                  stop_words='english')

  tf = tf_vectorizer.fit_transform(data)

  lda = LDA(n_components, niters=500)
  lda.fit(tf)
  lda.show_anime()
