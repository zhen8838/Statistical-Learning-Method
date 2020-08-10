from os import makedirs
import imageio
import os
from pathlib import Path
import imageio
from PIL import Image, ImageFont, ImageDraw
from typing import List
gifs = ['Perceptron', 'KNN', 'NaiveBayes', 'DecisionTree',
        'LogisticReression', 'SVM', 'Adaboost', 'ExpectationMaximization',
        'HMM_viterbi', 'crf', 'Kmeans', 'SVD',
        'PCA', 'LSA', 'PLSA', 'MCMC',
        'LDA', 'PageRank']
names = [
    '感知机', 'K近邻', '朴素贝叶斯', '决策树',
    '逻辑回归', '支持向量机', '提升方法', 'EM算法',
    '隐马尔可夫', '条件随机场', '聚类方法', '奇异值分解',
    '主成分分析', '潜在语义分析', '概率潜在语义分析', '马尔科夫链蒙特卡洛法',
    '潜在狄利克雷分配', 'PageRank',
]


def get_im_len(im: Image.Image) -> int:
  lens = 0
  if im.format == 'GIF':
    lens = im.n_frames - 1
  return lens


if __name__ == "__main__":
  ttf = "/usr/share/fonts/WinFonts/STSONG.TTF"
  font_size = 60
  font = ImageFont.truetype(ttf, font_size)

  ims = []
  for s, gif in zip(names, gifs):
    # 加载gif
    im = Image.open(f'gif/{gif}.gif' if gif != 'LSA' else f'gif/{gif}.png')
    # 设置片头
    title_im = Image.new('RGBA', im.size, color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(title_im)
    draw.text((320 - len(s) * font_size // 2,
               240 - font_size // 4 * 3),
              s, (0, 0, 0), font=font)
    for _ in range(10):
      ims.append(title_im.copy())

    # 设置内容
    lens = get_im_len(im)
    for i in range(40 if lens > 40 else lens + 1):
      im.seek(i)
      ims.append(im.copy())

  # Save the frames as an animated GIF
  ims[0].save('gif/merge.gif',
              save_all=True,
              append_images=ims[1:],
              duration=50,
              loop=0)
  # reduce gif file size
  # gifsicle -i gif/merge.gif -O3 --colors 256 -o  gif/merge.gif