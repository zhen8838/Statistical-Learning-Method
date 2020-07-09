import os
import sys
sys.path.insert(0, os.getcwd())
from engine import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import TruncatedSVD, PCA


class LSA(BaseModel):

  def fit(self, X, n_components):
    """

    Args:
        X (np.ndarray): [sample,text]

    Returns:

        T (np.ndarray): [sample,topic]

        V (np.ndarray): [topic,text] 
    """
    trucsvd = TruncatedSVD(n_components=n_components, random_state=10101)
    T = trucsvd.fit_transform(X)
    self.V = trucsvd.components_
    self.T = T
    return self.T, self.V

  def show_anime(self, Y, save_path='gif/LSA.png'):
    pca = PCA(n_components=3)
    dx = pca.fit_transform(self.T)

    fig: plt.Figure = plt.figure()
    fig.set_tight_layout(True)
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('Sample-topic')
    ax.view_init(elev=25., azim=120.)
    ax.scatter(dx[:, 0], dx[:, 1], dx[:, 2], c=Y)
    fig.savefig(save_path)
    plt.show()


if __name__ == "__main__":
  from sklearn.datasets import fetch_20newsgroups
  import nltk
  # nltk.download('stopwords')
  from nltk.corpus import stopwords
  stop_words = stopwords.words('english')
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.model_selection import train_test_split
  dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                               remove=('headers', 'footers', 'quotes'))
  documents = dataset.data

  clean_doc = map(lambda line: line.replace("[^a-zA-Z#]", " "), documents)
  clean_doc = map(lambda x: ' '.join([w for w in x.split() if len(w) > 3]), clean_doc)
  clean_doc = map(lambda x: x.lower(), clean_doc)
  tokenized_doc = map(lambda x: x.split(), clean_doc)
  tokenized_doc = map(lambda x: [item for item in x if item not in stop_words], tokenized_doc)
  detokenized_doc = map(lambda x: ' '.join(x), tokenized_doc)
  detokenized_doc = list(detokenized_doc)

  # keep top 1000 terms
  vectorizer = TfidfVectorizer(stop_words='english',
                               max_features=1000,
                               max_df=0.5,
                               smooth_idf=True)

  X = vectorizer.fit_transform(detokenized_doc)
  Y = dataset.target
  lsa = LSA()
  T, V = lsa.fit(X, 20)
  lsa.show_anime(Y)
  terms = vectorizer.get_feature_names()
  for i, comp in enumerate(V):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
    print("Topic ", str(i), ": ", ' '.join([t[0] for t in sorted_terms]))
