#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:48:55 2018

@author: kirstenbassuday
This program predicts language of input file
"""


import nltk
import re
import numpy as np
import math

filedir = '/Users/kirstenbassuday/Downloads'
pattern = "(\d+)|\"|\'|\“|\”|\‘|\`|\-|(\—)|\–|\...|\…|\+|\<|\<<|\»|\>|(\:)|\;|\’|\/|\.+|\,+|\:+|\!+|\*+|\%+|\&+|\?+|\€+|\£|\￡|\#+|\@+|\$+|\∞+|\§+|\||\[|\]|\(|\½|\)|\）|\{|\}|\•"

def preprocess(file_path):
    """Processes the corpus and remove patterns defined above"""
    corpus = []
    with open(file_path,mode ='r',encoding='UTF-8') as train_file:
        for line in train_file:
            line = line.lower()
            line = re.sub(r""+pattern, "", line) # remove digits
            corpus.append(line)
    return corpus


def train_language_model(file_path, lang_name):
    """Returns bigram and trigram language model"""
    sentence = ""
    for lang in languages:
        sentence = ''.join(preprocess(filedir+"/"+ file_path))
    #for bigram =2
    ngram= list(ngrams(sentence.split(), 2))
    ngram_len = len(ngram)
    ng_dist = nltk.FreqDist(ngram)
    ngrams_list =  sorted(ng_dist.items(), key=lambda item: item[1],reverse=True)
    ngram_model = dict(ngrams_list)
    ngram_model["NotFound"] = 0
    ngram_model = {a: (b + 1) / (ngram_len + len(ngram_model)) for a,b in ngram_model.items()}
    
    #for trigram = 3
    trigram= list(ngrams(sentence.split(), 3))
    trigram_len = len(trigram)
    tri_dist = nltk.FreqDist(trigram)
    trigrams_list =  sorted(tri_dist.items(), key=lambda item: item[1],reverse=True)
    trigram_model = dict(trigrams_list)
    trigram_model["NotFound"] = 0
    trigram_model = {a: (b + 1) / (trigram_len + len(trigram_model)) for a,b in trigram_model.items()}
    return ngram_model,trigram_model


def test_language_model(file_path, lang_name,modelList):
    """Prints the result of the tested langauge against the trained models"""
    bigramList = []
    sumprob =0 #total prob for the test file
    acc = 0
    each_sentence_prob =[]
    probList =[] # list of all prob for test lang against each model
    model_name = ["french","english","german","italian","dutch","spanish"]
    guessedList =[]

      #read the test file
    test_file = preprocess(file_path)
    for sentence in test_file:
        ngram = nltk.word_tokenize(sentence)
        # build the n-gram models
        bigram = nltk.bigrams(ngram)
        test = list(bigram)
       # print(test)
       # print("---------")
        for model in modelList:
            for mytuple in model:
                a,b = mytuple
                for item in test:
                    if item == a:
                        prob = (b/len(model))
                        sumprob += math.log(prob)
                    each_sentence_prob.append(sumprob)
                    print("test 1",each_sentence_prob)
        maxprob = max(each_sentence_prob)
        indexOfmaxProb = each_sentence_prob.index(maxprob)
        print("test2",each_sentence_prob)
        #guessedList.append(model_name[indexOfmaxProb])
    #print(guessedList)
    return bigramList
    

def corpus_statistics(lang_name,langpath):
    """Prints the corpus statistics of the bigrams and trigrams in the training model function"""
    test_file = filedir+"/"+"test/"+"/"+langpath    
    train_file = filedir+"/"+"train/"+"/"+langpath  
    
    
    train_sentence_count = len(preprocess(train_file))
    corpus_count = len(preprocess(test_file))
    corpus = ''.join(preprocess(train_file))
    
    bigram = list(ngrams(corpus.split(), 2))
    trigram = list(ngrams(corpus.split(), 3))
    
    print("Corpus statistics")
    print("--------------")
    print("Language Name:", lang_name)
    print("----------------------")
    print("Training sentence:", train_sentence_count)
    print("Testing Sentence:", corpus_count)
    
    print("Train Number of bi-grams :", len(bigram))
    
    print("Train Number of tri-grams :", len(trigram))
    print()

if __name__ == "__main__":
    print("""Assignment 2: Language Identification 
    Name: Kirsten Bassuday """)
    print("Language Prediction")
    print("--------------")
    lang_name = ["french","english","german","italian","dutch","spanish"]
    lang_path = ["french/french.txt","english/english.txt","dutch/dutch.txt","italian/italian.txt","germany/germany.txt","spanish/spanish.txt"]    
    models =[]
    
    for i,file_name in enumerate(lang_path):
        bigramlist,trigramlist = train_language_model("train/"+file_name, lang_name[i])
        corpus_statistics(lang_name[i],file_name)
        
   # for i,file_name in enumerate(lang_path):
    #    test_language_model(filedir+"/test/"+file_name,lang_name[i],bigramlist)


