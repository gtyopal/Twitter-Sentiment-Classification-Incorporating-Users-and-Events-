
# coding: utf-8

import gensim
import numpy as np
import pandas as pd
from sklearn import svm
from collections import Counter
import os

#return all the tokens of sentence according grams
def tokenize(sentence, grams):
      words = sentence.split()
      tokens = []
      for gram in grams:
          for i in range(len(words) - gram + 1):
              tokens += ["_*_".join(words[i:i+gram])]
      return tokens

#Input: file
#Output:Dict{token, count_number_in_file} 
def build_dict(filename, grams):
    dic = Counter()
    with open(filename,'r') as f:
        for sentence in f:
            dic.update(tokenize(sentence, grams))
    return dic 

def process_files(file_pos, file_neg, dic, r, outfn, grams):
    output = []
    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        for l in open(f).readlines():
            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
               
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()

#p:propobility of each token in positive sample
#q:propobility of each token in negative sample
#r:log(p/q) elmentwise
#dic:{token， index}
def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set( list(poscounts.keys()) +  list(negcounts.keys())))
    #dic: token -----> index
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print("computing ...")
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r

ptrain = "../word2vec/ptrain-norm.txt"
ntrain = "../word2vec/ntrain-norm.txt"
ptest = "../word2vec/ptest-norm.txt"
ntest = "../word2vec/ntest-norm.txt"
liblinear = "./liblinear-2.1/" # you need to have gcc in your computer and compile the "Makefile" in liblinear-2.1 folder to use liblinear

def main(ptrain, ntrain, ptest, ntest, liblinear, ngram):
    ngram = [int(i) for i in ngram]
    print("counting...")
    poscounts = build_dict(ptrain, ngram)    
    negcounts = build_dict(ntrain, ngram)    
    
    dic, r = compute_ratio(poscounts, negcounts)
    print("processing files...")
    process_files(ptrain, ntrain, dic, r, "train-nbsvm.txt", ngram)
    process_files(ptest, ntest, dic, r, "test-nbsvm.txt", ngram)

    trainsvm = os.path.join(liblinear, "train")
    predictsvm = os.path.join(liblinear, "predict")
    print("training model...")
    os.system(trainsvm + " -s 2 train-nbsvm.txt model.liblinear")
    print("predict...")
    os.system(predictsvm + " -b 0 test-nbsvm.txt model.liblinear result.txt")

def compute_acc(predict_file, test_file):
    with open(test_file) as tf:
        y_ = [ int(line.split()[0]) for line in tf]
    with open(predict_file) as pf:
        y = [ int(line) for line in pf]
    result = np.equal(y, y_).astype(int)
    return result.mean()


if __name__ == "__main__":
    main(ptrain, ntrain, ptest, ntest, liblinear, "1")
    acc = compute_acc("result.txt", "test-nbsvm.txt")
    print("accuracy:%f" % (acc))







