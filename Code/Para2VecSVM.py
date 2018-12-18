# coding: utf-8

import gensim
from gensim.models.doc2vec import TaggedLineDocument
import numpy as np
import pandas as pd
from sklearn import svm

#count number of words in vocabulary from word2vec in a sentence
#Input: a sentence 
#Output: number
def countWordInVocab(sentence):
    num = 0
    vocab = word2vec_model.vocab
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
    test_deleted = 0
    for index, sen in enumerate(sentences):
        if countWordInVocab(sen) == 0:
            if index < 800000:
                neg_deleted += 1
            elif index < 1600000:
                pos_deleted += 1
            else:
                test_deleted += 1
        else:
            pass
    return 800000-neg_deleted, 800000-pos_deleted, 359-test_deleted

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
    global X_train, y_train, X_test, y_test
    
    sentences = gensim.models.word2vec.LineSentence("../word2vec/trainAndTest.csv")
    neg_num, pos_num, test_num = cntNumNegPos(sentences)
    print(neg_num, pos_num, test_num)
    doc2vecCorpus = deleteBlankSen(sentences)
        finalCorpus = [ " ".join(sen)  for sen in doc2vecCorpus]
    with open("doc2vecCorpus.txt", 'w') as outfile:
        outfile.write("\n".join(finalCorpus))
    
    documents = TaggedLineDocument("doc2vecCorpus.txt")
    doc2vec_model = gensim.models.Doc2Vec(documents, size=100, window=5, min_count=5, 
                                          workers=4, dm_mean=1, dbow_words=1)
    ##Persist a model to disk with:
    #model.save("doc2vec.model"
    #model = gensim.models.Doc2Vec.load("doc2vec.model")  # you can continue training with the loaded model!
    train_index = [index for index in range(neg_num+pos_num)]
    X_train = model.docvecs[train_index]
    y_train = [0] * neg_num + [4] * pos_num
    test_index = [ index+neg_num+pos_num for index in range(test_num)]
    X_test = model.docvecs[test_index]

    testData = pd.read_csv("../word2vec/testdata.csv", encoding="latin-1", names=["label","text"])
    test_sentences = gensim.models.word2vec.LineSentence("../word2vec/test_text-norm.csv")
    y_test = list(testData['label'])



# train SVM 
def train():
    global X_train, y_train, X_test, y_test
    clf = svm.LinearSVC(dual=False)
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
    word2vec_model = gensim.models.Word2Vec.load("word2vec.model") 
    main()






