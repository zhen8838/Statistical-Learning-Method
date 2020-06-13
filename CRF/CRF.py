""" 参考自https://github.com/bojone/crf/ """
import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend
from sklearn.model_selection import train_test_split
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CRF(kl.Layer):
  """
  CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
  而预测则需要另外建立模型。
  """

  def __init__(self, ignore_last_label=False, lr_mult=1., **kwargs):
    """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
    """
    super().__init__(**kwargs)
    self.ignore_last_label = 1 if ignore_last_label else 0
    self.lr_mult = lr_mult

  def build(self, input_shape):
    self.num_labels = input_shape[-1] - self.ignore_last_label
    self._trans: tf.Variable = self.add_weight(name='crf_trans',
                                               shape=(self.num_labels, self.num_labels),
                                               initializer='glorot_uniform',
                                               trainable=True)
    self._trans.assign(self._trans / self.lr_mult)

    self.trans = lambda: self._trans * self.lr_mult

  def get_weights(self):
    weights = super().get_weights()
    return [w * self.lr_mult for w in weights]

  def log_norm_step(self, inputs, states):
    """递归计算归一化因子
    要点：1、递归计算；2、用logsumexp避免溢出。
    技巧：通过expand_dims来对齐张量。
    """
    inputs, mask = inputs[:, :-1], inputs[:, -1:]
    states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
    trans = K.expand_dims(self.trans(), 0)  # (1, output_dim, output_dim)
    outputs = tf.math.reduce_logsumexp(states + trans, 1)  # (batch_size, output_dim)
    outputs = outputs + inputs
    outputs = mask * outputs + (1 - mask) * states[:, :, 0]
    return outputs, [outputs]

  def path_score(self, inputs, labels):
    """计算目标路径的相对概率（还没有归一化）
    要点：逐标签得分，加上转移概率得分。
    技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
    """
    point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分
    labels1 = K.expand_dims(labels[:, :-1], 3)
    labels2 = K.expand_dims(labels[:, 1:], 2)
    labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
    trans = K.expand_dims(K.expand_dims(self.trans(), 0), 0)
    trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
    return point_score + trans_score  # 两部分得分之和

  def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
    return inputs

  def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
    if self.ignore_last_label:
      mask = 1 - y_true[:, :, -1:]
    else:
      mask = K.ones_like(y_pred[:, :, :1])
    y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
    path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
    init_states = [y_pred[:, 0]]  # 初始状态
    y_pred = K.concatenate([y_pred, mask])
    log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
    log_norm = tf.math.reduce_logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
    return log_norm - path_score  # 即log(分子/分母)

  def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
    mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
    y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
    isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
    isequal = K.cast(isequal, 'float32')
    if mask == None:
      return K.mean(isequal)
    else:
      return K.sum(isequal * mask) / K.sum(mask)


def max_in_dict(d):  # 定义一个求字典中最大值的函数
  dict_items = list(d.items())
  key, value = dict_items[0]
  for i, j in dict_items[1:]:
    if j > value:
      key, value = i, j
  return key, value


def viterbi(nodes, trans):  # viterbi算法，跟前面的HMM一致
  paths = nodes[0]  # 初始化起始路径
  for l in range(1, len(nodes)):  # 遍历后面的节点
    paths_old, paths = paths, {}
    for n, ns in nodes[l].items():  # 当前时刻的所有节点
      max_path, max_score = '', -1e10
      for p, ps in paths_old.items():  # 截止至前一时刻的最优路径集合
        score = ns + ps + trans[p[-1] + n]  # 计算新分数
        if score > max_score:  # 如果新分数大于已有的最大分
          max_path, max_score = p + n, score  # 更新路径
      paths[max_path] = max_score  # 储存到当前时刻所有节点的最优路径
  return max_in_dict(paths)


def cut(s, trans, char2id):  # 分词函数，也跟前面的HMM基本一致
  if not s:  # 空字符直接返回
    return []
  # 字序列转化为id序列。注意，经过我们前面对语料的预处理，字符集是没有空格的，
  # 所以这里简单将空格的id跟句号的id等同起来
  sent_ids = np.array([[char2id.get(c, 0) if c != ' ' else char2id[u'。']
                        for c in s]])
  probas = model.predict(sent_ids)[0]  # [n,5]
  nodes = [dict(zip('sbme', i)) for i in probas[:, :4]]  # 只取前4个,因为最后一个是mask
  nodes[0] = {i: j for i, j in nodes[0].items() if i in 'bs'}  # 首字标签只能是b或s
  nodes[-1] = {i: j for i, j in nodes[-1].items() if i in 'es'}  # 末字标签只能是e或s
  tags = viterbi(nodes, trans)[0]
  result = [s[0]]
  for i, j in zip(s[1:], tags[1:]):
    if j in 'bs':  # 词的开始
      result.append(i)
    else:  # 接着原来的词
      result[-1] += i
  return result


class Evaluate(k.callbacks.Callback):
  def __init__(self, tag2id, char2id):
    self.highest = 0.
    self.tag2id = tag2id
    self.char2id = char2id
    self.history = []

  def on_train_batch_end(self, batch, logs=None):
    A = self.model.get_layer('crf').get_weights()[0][:4, :4]  # 从训练模型中取出最新得到的转移矩阵
    self.history.append(A)

  # def on_epoch_end(self, epoch, logs=None):
  #   A = self.model.get_weights()[-1][:4, :4]  # 从训练模型中取出最新得到的转移矩阵
  #   trans = {}
  #   for i in 'sbme':
  #     for j in 'sbme':
  #       trans[i + j] = A[self.tag2id[i], self.tag2id[j]]
  #   right = 0.
  #   total = 0.
  #   for s in tqdm(iter(valid_sents), desc=u'验证模型中'):
  #     result = cut(''.join(s), trans, self.char2id)
  #     total += len(set(s))
  #     right += len(set(s) & set(result))  # 直接将词集的交集作为正确数。该指标比较简单，
  #     # 也许会导致估计偏高。读者可以考虑自定义指标
  #   acc = right / total
  #   if acc > self.highest:
  #     self.highest = acc
  #   print('val acc: %s, highest: %s' % (acc, self.highest))

  def show_anime(self, save_path='gif/crf.gif'):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax: plt.Axes
    A = self.history[0]
    c = ax.pcolor(A, cmap='RdBu_r', vmin=A.min(), vmax=A.max(),
                  edgecolors='w', linewidths=30)
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_xticklabels(list('sbme'))
    ax.set_yticklabels(list('sbme'))
    for i in range(4):
      for j in range(4):
        text = ax.text(j + 0.5, i + 0.5,
                       f'{A[i, j]:^4.2f}',
                       ha="center", va="center", color="w")

    def update(t):
      ax.cla()
      ax.set_title(f'iter {t}')
      ax.set_xticks(np.arange(4) + 0.5)
      ax.set_yticks(np.arange(4) + 0.5)
      ax.set_xticklabels(list('sbme'))
      ax.set_yticklabels(list('sbme'))
      A = self.history[t]
      c = ax.pcolor(A, cmap='RdBu_r', vmin=A.min(), vmax=A.max(),
                    edgecolors='w', linewidths=30)

      for i in range(4):
        for j in range(4):
          text = ax.text(j + 0.5, i + 0.5,
                         f'{A[i, j]:^4.2f}',
                         ha="center", va="center", color="w")

    anim = FuncAnimation(fig, update, frames=len(self.history), interval=100)
    anim.save(save_path, writer='imagemagick', fps=5)
    plt.show()


if __name__ == "__main__":
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  sents = []
  with open('CRF/msr_training.utf8', 'r') as f:
    for line in f.readlines():
      sents.append(line.strip())
  sents = [re.split(' +', s) for s in sents]  # 词之间以空格隔开
  sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
  np.random.shuffle(sents)  # 打乱语料，以便后面划分验证集

  chars = {}  # 统计字表
  for s in sents:
    for c in ''.join(s):
      if c in chars:
        chars[c] += 1
      else:
        chars[c] = 1
  # 过滤低频字
  min_count = 2
  chars = {i: j for i, j in chars.items() if j >= min_count}
  id2char = {i + 1: j for i, j in enumerate(chars)}  # id到字的映射
  char2id = {j: i for i, j in id2char.items()}  # 字到id的映射

  id2tag = {0: 's', 1: 'b', 2: 'm', 3: 'e'}  # 标签（sbme）与id之间的映射
  tag2id = {j: i for i, j in id2tag.items()}

  train_sents, valid_sents = train_test_split(sents, test_size=0.05)

  batch_size = 128

  def train_generator():
    while True:
      X, Y = [], []
      for i, s in enumerate(train_sents):  # 遍历每个句子
        sx, sy = [], []
        for w in s:  # 遍历句子中的每个词
          sx.extend([char2id.get(c, 0) for c in w])  # 遍历词中的每个字
          if len(w) == 1:
            sy.append(0)  # 单字词的标签
          elif len(w) == 2:
            sy.extend([1, 3])  # 双字词的标签
          else:
            sy.extend([1] + [2] * (len(w) - 2) + [3])  # 多于两字的词的标签
        X.append(sx)
        Y.append(sy)
        if len(X) == batch_size or i == len(train_sents) - 1:  # 如果达到一个batch
          maxlen = max([len(x) for x in X])  # 找出最大字数
          X = [x + [0] * (maxlen - len(x)) for x in X]  # 不足则补零
          Y = [y + [4] * (maxlen - len(y)) for y in Y]  # 不足则补第五个标签
          yield np.array(X), tf.keras.utils.to_categorical(Y, 5)
          X, Y = [], []

  embedding_size = 128
  sequence = kl.Input(shape=(None,), dtype='int32')  # 建立输入层，输入长度设为None
  embedding = kl.Embedding(len(chars) + 1, embedding_size)(sequence)  # 去掉了mask_zero=True
  cnn = kl.Conv1D(128, 3, activation='relu', padding='same')(embedding)
  cnn = kl.Conv1D(128, 3, activation='relu', padding='same')(cnn)
  cnn = kl.Conv1D(128, 3, activation='relu', padding='same')(cnn)  # 层叠了3层CNN

  crf = CRF(True, lr_mult=100.)  # 定义crf层，参数为True，自动mask掉最后一个标签,同时增大crf学习率100倍
  tag_score = kl.Dense(5)(cnn)  # 变成了5分类，第五个标签用来mask掉
  tag_score = crf(tag_score)  # 包装一下原来的tag_score

  model = k.Model(inputs=sequence, outputs=tag_score)
  model.summary()

  model.compile(loss=crf.loss,  # 用crf自带的loss
                optimizer=k.optimizers.Adam(0.001),
                metrics=[crf.accuracy]  # 用crf自带的accuracy
                )
  evaluator = Evaluate(tag2id, char2id)
  model.fit_generator(train_generator(),
                      steps_per_epoch=100,
                      epochs=1,
                      callbacks=[evaluator])  # 训练并将evaluator加入到训练过程
  A = model.get_layer('crf').get_weights()[0][:4, :4]  # :4是为了去除mask的转义概率
  trans = {}
  for i in 'sbme':
    for j in 'sbme':
      trans[i + j] = A[tag2id[i], tag2id[j]]
  right = 0.
  total = 0.
  for s in range(5):
    s = valid_sents[s]
    result = cut(''.join(s), trans, char2id)
    print(''.join(s), '\n', result)

  evaluator.show_anime()

