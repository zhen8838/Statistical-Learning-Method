import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from engine import BaseModel, Dict
from sklearn.preprocessing import normalize


class HMM(BaseModel):
  def __init__(self, A, B, pi):
    super().__init__()
    self.A = A  # 状态转移矩阵 NxN ，A[i,j] = P(t+1时刻状态j|t时刻状态i)
    self.B = B  # 观测概率矩阵 NxM ，B[i,j] = P(t时刻观测j|t时刻状态i)
    self.pi = pi  # 初始概率矩阵 N

  def init_param(self, O):
    self.T = len(O)
    self.N, self.M = self.B.shape
    self.N: int
    self.M: int

  def forward(self, O):
    self.init_param(O)
    self.alpha: np.ndarray = np.zeros((self.T, self.N))
    for t in range(self.T):
      if t == 0:
        # NOTE B.T is [M,N] -> B.T[O[t]] is [N]
        self.alpha[t] = self.pi * self.B.T[O[t]]
      else:
        # NOTE 取a_ji,因此这里进行转置
        self.alpha[t] = np.sum(self.alpha[t - 1] * self.A.T, -1) * self.B.T[O[t]]
    return np.sum(self.alpha[t])

  def backward(self, O):
    self.init_param(O)
    self.beta: np.ndarray = np.zeros((self.T, self.N))
    for t in range(self.T)[::-1]:
      if t == self.T - 1:
        self.beta[t] = 1.
      else:
        # NOTE 这里取a_ij因此不用转置
        self.beta[t] = np.sum(self.beta[t + 1] * self.A * self.B.T[O[t + 1]], -1)
    return np.sum(self.pi * self.B.T[O[0]] * self.beta[0])

  def baum_welch(self):
    raise NotImplementedError('Hope some one can finish this algorithm')

  def viterbi(self, O):
    self.init_param(O)
    self.sigma: np.ndarray = np.zeros((self.T, self.N))
    self.delta: np.ndarray = np.zeros((self.T, self.N), dtype=np.uint8)
    for t in range(self.T):
      if t == 0:
        self.sigma[t] = self.pi * self.B.T[O[t]]
      else:
        self.sigma[t] = np.max(self.sigma[t - 1] * self.A.T, -1) * self.B.T[O[t]]
        self.delta[t] = np.argmax(self.sigma[t - 1] * self.A.T, -1)

    maxP = np.max(self.sigma[-1])
    self.I = np.zeros(self.T, dtype=np.uint8)
    for t in range(self.T)[::-1]:
      if t == self.T - 1:
        self.I[t] = np.argmax(self.sigma[t])
      else:
        self.I[t] = self.delta[t + 1, self.I[t + 1]]

  def show_anime(self, save_path='gif/HMM_viterbi.gif'):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes

    ax.set_title('HMM viterbi')
    ax.set_xlabel('states')
    ax.set_ylabel('times')

    def update(t):
      if t == 0:
        c = ax.pcolor(normalize(self.sigma), edgecolors='w', linewidths=30)
        for t in range(self.T):
          for n in range(self.N):
            text = ax.text(n + 0.5, t + 0.5, f'{self.sigma[t, n]:.5f}',
                           ha="center", va="center", color="r")
      else:
        t = self.T - t
        for n in range(self.N):
          last = self.delta[t][n]
          ax.arrow(last + .5, t - .5,
                   n + .5 - (last + .5),
                   t + .5 - (t - .5),
                   head_width=0.05, head_length=0.1,
                   fc='k', ec='r' if self.I[t] == n else 'k')
    anim = FuncAnimation(fig, update, frames=self.T, interval=500)
    anim.save(save_path, writer='imagemagick', fps=1)
    plt.show()


if __name__ == "__main__":
  A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
  B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
  pi = np.array([0.2, 0.4, 0.4])
  O = np.array([0, 1, 0])

  hmm = HMM(A, B, pi)
  print(hmm.forward(O))
  print(hmm.backward(O))
  hmm.viterbi(O)
  hmm.show_anime()
