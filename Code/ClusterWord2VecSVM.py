# coding: utf-8

import gensim
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#load word2vec model trained 
model = gensim.models.Word2Vec.load("word2vec.model")

index2vec = model[model.index2word]
index2word = model.index2word

np.random.seed(5)
estimators = KMeans(n_clusters=200, init='k-means++', precompute_distances=True, n_jobs=-1)
estimators.fit(index2vec)

labels = estimators.labels_
cluster2vec = estimators.cluster_centers_
word2cluster = { index2word[index]:cluster for  index, cluster in enumerate(labels) }



#build training data X
#Input: a sentence 
#Output: vector of the sentence using average of cluster_center of words
def ClusterWord2Vec(sentence, vec_size=100):
    global word2cluster, cluster2vec, miss_num
    vector = np.zeros(vec_size)
    num = len(sentence)
    for word in sentence:
        try:
            cluster = word2cluster[word]
            vector += cluster2vec[cluster]
        except KeyError:
            num -= 1
    if num > 0:
        return list(vector / num)
    else:
        return list(vector)



#count number of words in vocabulary from word2vec in a sentence
#Input: a sentence 
#Output: number
def countWordInVocab(sentence):
    num = 0
    vocab = model.vocab
    for word in sentence:
        if word in vocab:
            num += 1
        else:
            pass
    return num

#count number of neg and pos to be deleted
#Input: sentences
#Output: neg_number, pos_number
def cntNumNegPos(sentences):
    neg_deleted = 0
    pos_deleted = 0
    for index, sen in enumerate(sentences):
        if countWordInVocab(sen) == 0:
            if index < 800000:
                neg_deleted += 1
            else:
                pos_deleted += 1
        else:
            pass
    return 800000-neg_deleted, 800000-pos_deleted

#delete blank sentence
#Input:sentences
#Output: processed sentences
#parameter must be mutable, then can be changed in function
def deleteBlankSen(sentences):
    return [sen for sen in sentences if countWordInVocab(sen) > 0]


X_train = []
y_train = []
X_test = []
y_test = []

def prepareData():
    global X_train, y_train, X_test, y_test, word2cluster, cluster2vec
    #sentences = gensim.models.word2vec.LineSentence("../word2vce/train_text-norm")
    sentences = gensim.models.word2vec.LineSentence("../word2vec/train_text-norm.csv")
    neg_num, pos_num = cntNumNegPos(sentences)
    X_train =   list(map(ClusterWord2Vec, deleteBlankSen(sentences)))  
    y_train = [0] * neg_num + [4] * pos_num
    #y_train = [0] * 50 + [4] * 50
    testData = pd.read_csv("../word2ved/testdata.csv", encoding="latin-1", names=["label","text"])
    #test = testData[testData['label'] != 2]
    #test.index = range(len(test))
    test_sentences = gensim.models.word2vec.LineSentence("../word2vec/test_text-norm.csv")
    X_test = list(map(ClusterWord2Vec, test_sentences))
    y_test = list(testData['label'])


# train SVM 
def train():
    global X_train, y_train, X_test, y_test
    #clf = svm.SVC(kernel='linear')
    clf = svm.LinearSVC(dual=False)
    clf.fit(X_train, y_train)
    #print("number of support vector: %d\t%d" %(clf.n_support_[0], clf.n_support_[1]))
    return clf
    #print SVM information
    #clf.n_support_
    #clf.dual_coef_
    #clf.coef_[0]
    
def compute_accuracy(y, y_):
    result = np.equal(y, y_).astype(int)
    return result.mean()

def main():
    global y_test
    prepareData()
    #train the SVM model
    clf = train()
    #predict
    y = clf.predict(X_test)
    acc = compute_accuracy(y, y_test)
    print("accuracy:%f" % (acc))


if __name__ == "__main__":
    main()






