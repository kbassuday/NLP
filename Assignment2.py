#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:06:09 2018

@author: kirstenbassuday
"""


from __future__ import print_function, division
from io import open

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lecture2 import PegasosWithLogLoss
from lecture2 import PegasosWithSVC



# This function reads the corpus, returns a list of documents, and a list
# of their corresponding polarity labels.
def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            X.append(words[3:])
            Y.append(words[1])
    return X, Y


if __name__ == '__main__':

    # Read all the documents.
    X, Y = read_data('all_sentiment_shuffled.txt')

    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

    # Set up the preprocessing steps and the classifier.
    myclf=PegasosWithSVC()
    #please uncomment to run the below log loss and then comment the above 
    #myclf=PegasosWithLogLoss()
    model_pl = make_pipeline(
        TfidfVectorizer(preprocessor = lambda x: x, tokenizer = lambda x: x),
        SelectKBest(k=1000),
        Normalizer(),
        myclf,
    )

    t0 = time.time()
    model_pl.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    Yguess = model_pl.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))
