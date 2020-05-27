from abc import abstractmethod
import numpy as np


class BaseModel(object):
  def __init__(self):
    super().__init__()

  @abstractmethod
  def call(self):
    return NotImplementedError

  @abstractmethod
  def update(self):
    return NotImplementedError

  @abstractmethod
  def fit(self, X, Y, epoch, lr):
    return NotImplementedError

  @abstractmethod
  def predict(self, test_x):
    return NotImplementedError

  def show_anime(self, save_path: str):
    self.save_path = save_path
    pass

  @abstractmethod
  def on_train_begin(self):
    pass

  @abstractmethod
  def on_train_batch_begin(self):
    pass

  @abstractmethod
  def on_train_batch_end(self):
    pass

  @abstractmethod
  def on_train_epoch_begin(self):
    pass

  @abstractmethod
  def on_train_epoch_end(self):
    pass

  @abstractmethod
  def on_train_begin(self):
    pass


class Dict(dict):
  """ warper for dict, support numpy getitem """

  def __getitem__(self, key):
    if isinstance(key, np.ndarray):
      l = []
      for k in key:
        l.append(super().__getitem__(k))
      return np.array(l)
    else:
      return super().__getitem__(key)
