# coding: utf-8


import gensim
import numpy as np
import pandas as pd
from sklearn import svm
import logging



# trian "word2vec.model" to be used
sentences = gensim.models.word2vec.LineSentence("../word2vec/train_text-norm.csv")
model = gensim.models.Word2Vec(sentences,sg=1, size=500, window=5,sample=1e-5,hs=0,negative=5, min_count=5, worker=4)
model.save("word2vec.model")




