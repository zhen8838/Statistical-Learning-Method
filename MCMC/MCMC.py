import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
plt.style.use('ggplot')


class Gibbs(BaseModel):
  def __init__(self, niters: int, burnin: int):
    super().__init__()
    self.niters = niters
    self.burnin = burnin
    self.history = []

  def proposal(self, theta):
    rho = 0.5
    m1, m2 = 1, 3
    s1, s2 = 3, 2

    def p_y_x(x):
      return norm(m2 + rho * s2 / s1 * (x - m1),
                  np.sqrt(1 - rho**2) * s2).rvs()

    def p_x_y(y):
      return norm(m1 + rho * s1 / s2 * (y - m2),
                  np.sqrt(1 - rho**2) * s1).rvs()

    theta = [p_y_x(theta[1]), theta[1]]
    self.history.append(theta.copy())
    theta = [theta[0], p_x_y(theta[0])]
    self.history.append(theta.copy())
    return theta

  def fit(self, theta: np.ndarray):
    thetas = np.zeros((self.niters - self.burnin, 2), np.float)
    for i in range(self.niters):
      theta = self.proposal(theta)
      if i >= self.burnin:
        thetas[i - self.burnin] = theta
    self.history = np.array(self.history)
    return thetas

  def show_anime(self, save_path='gif/MCMC.gif'):
    fig: plt.Figure = plt.figure()
    fig.suptitle('Gibbs Sampling')
    fig.set_tight_layout(True)
    true_m1 = 4
    true_m2 = 3
    samples = 250
    samples_width = (0, samples)
    i_width = (true_m1 - 10, true_m1 + 10)
    s_width = (true_m2 - 10, true_m2 + 10)
    ax1 = fig.add_subplot(221, xlim=i_width, ylim=samples_width)
    ax2 = fig.add_subplot(224, xlim=samples_width, ylim=s_width)
    ax3 = fig.add_subplot(223, xlim=i_width, ylim=s_width,
                          xlabel='a',
                          ylabel='b')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    linea, = ax1.plot([], [], lw=1)
    lineb, = ax2.plot([], [], lw=1)
    new_points, = ax3.plot([], [], 'o', lw=2, alpha=.6)
    old_points, = ax3.plot([], [], lw=1, alpha=.3)
    center_x, = ax3.plot([], [], 'k', lw=1)
    center_y, = ax3.plot([], [], 'k', lw=1)
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    lines = [linea, lineb, new_points,
             old_points, center_x, center_y]

    def init():
      for line in lines:
        line.set_data([], [])
      return lines

    def animate(i):
      if i < 1:
        return lines
      linea.set_data(self.history[:i, 0] if i < samples
                     else self.history[i - samples:i, 0],
                     np.arange(i) if i < samples
                     else np.arange(samples))
      lineb.set_data(np.arange(i) if i < samples
                     else np.arange(samples),
                     self.history[:i, 1] if i < samples
                     else self.history[i - samples:i, 1])
      new_points.set_data(self.history[:i, 0], self.history[:i, 1])
      old_points.set_data(self.history[:i, 0], self.history[:i, 1])
      x, y = self.history[i - 1]
      center_x.set_data([x, x], [y, s_width[1]])
      center_y.set_data([x, i_width[1]], [y, y])
      return lines

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(self.history), interval=50, blit=True)
    anim.save(save_path, writer='imagemagick', fps=30)
    plt.show()


if __name__ == "__main__":
  gibbs = Gibbs(500, 100)
  thetas = gibbs.fit([0.2, 0.3])
  gibbs.show_anime()
  plt.hist(thetas[:, 0], 50, density=True)
  plt.hist(thetas[:, 1], 50, density=True)
  plt.show()
