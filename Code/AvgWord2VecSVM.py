# coding: utf-8

import gensim
import numpy as np
import pandas as pd
from sklearn import svm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print(mkl_get_max_threads())



#load word2vec model trained 
model = gensim.models.Word2Vec.load("word2vec.model")

X_train = []
y_train = []
X_test = []
y_test = []


#build training data X
#Input: a sentence 
#Output: vector of the sentence using average of words
def AvgWord2Vec(sentence, vec_size=100):
    vector = np.zeros(vec_size)
    num = len(sentence)
    for word in sentence:
        try:
            vector += model[word]
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

    
def prepareData():
    global X_train, y_train, X_test, y_test
    sentences = gensim.models.word2vec.LineSentence("../word2vec/train_text-norm.csv")
    
    neg_num, pos_num = cntNumNegPos(sentences)
    X_train =   list(map(AvgWord2Vec, deleteBlankSen(sentences)))  
    y_train = [0] * neg_num + [4] * pos_num

    
    testData = pd.read_csv("../word2vec/testdata.csv", encoding="latin-1", names=["label","text"])

    test_sentences = gensim.models.word2vec.LineSentence("../word2vec/test_text-norm.csv")
    X_test = list(map(AvgWord2Vec, test_sentences))
    y_test = list(testData['label'])
    
# train SVM 
def train():
    global X_train, y_train, X_test, y_test
    clf  = svm.LinearSVC(dual=False)
    clf.fit(X_train, y_train)

    return clf

    
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

