  # !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:32:12 2018

@author: kirstenbassuday
This program takes tokenized corpus and prints its stats, bigrams, trigrams
"""

import nltk
import matplotlib.pyplot as plt
import re
from collections import Counter

# constants

tokenize_re = re.compile(r"""([n]\'[t])|(\'[re]+)|(Mrs?\.)|\w+\,\w+\,\w+|(\w+\-\w+\-?\w*)|([A-Z]\.[A-Z]\.)|([A-Z]\.)|([A-Z][a-z]{1,3}\s?\.)|('\w)|(\d+[\,|.]?\d+)|(\d+\/\d+)|(\d+[s]?)|(\w)+ |(\,)|(\")|(\.)|(\.\.\.)|(\')|(\--)|(\:)|(\;)|(\$)|(\&)|(\?)|(\%)|(\{)|(\})|\)|\(|(\#)|(\!)""",re.VERBOSE)


# function/class definitions
def get_corpus_text(nr_files):
    """Returns the raw corpus as a long string.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_raw.fileids()[:nr_files]
    corpus_text = nltk.corpus.treebank_raw.raw(fileids)
    # Get rid of the ".START" text in the beginning of each file:
    corpus_text = corpus_text.replace(".START", "")
    return corpus_text


def fix_treebank_tokens(tokens):
    """Replace tokens so that they are similar to the raw corpus text."""
    return [token.replace("''", '"').replace("``", '"').replace(r"\/", "/")
            for token in tokens]


def get_gold_tokens(nr_files):
    """Returns the gold corpus as a list of strings.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_chunk.fileids()[:nr_files]
    gold_tokens = nltk.corpus.treebank_chunk.words(fileids)

    return fix_treebank_tokens(gold_tokens)


def tokenize_corpus(text):
    """Performs tokenization"""
    return [m.group() for m in tokenize_re.finditer(text)]


def evaluate_tokenization(test_tokens, gold_tokens):
    """Finds the chunks where test_tokens differs from gold_tokens.
    Prints the errors and calculates similarity measures.
    """
    import difflib
    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(test_tokens, gold_tokens)
    error_chunks = true_positives = false_positives = false_negatives = 0
    print(" Token%30s  |  %-30sToken" % ("Error", "Correct"))
    print("-" * 38 + "+" + "-" * 38)
    for difftype, test_from, test_to, gold_from, gold_to in matcher.get_opcodes():
        if difftype == "equal":
            true_positives += test_to - test_from
        else:
            false_positives += test_to - test_from
            false_negatives += gold_to - gold_from
            error_chunks += 1
            test_chunk = " ".join(test_tokens[test_from:test_to])
            gold_chunk = " ".join(gold_tokens[gold_from:gold_to])
            print("%6d%30s  |  %-30s%d" % (test_from, test_chunk, gold_chunk, gold_from))
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    fscore = 2.0 * precision * recall / (precision + recall)
    print()
    print("Test size: %5d tokens" % len(test_tokens))
    print("Gold size: %5d tokens" % len(gold_tokens))
    print("Nr errors: %5d chunks" % error_chunks)
    print("Precision: %5.2f %%" % (100 * precision))
    print("Recall:    %5.2f %%" % (100 * recall))
    print("F-score:   %5.2f %%" % (100 * fscore))
    print()

def nr_corpus_words(corpus):
    '''Returns number of word tokens and word types'''
    nr_of_corpus_words = len(corpus)
    wordtypes = len(set(corpus))
    return nr_of_corpus_words, wordtypes

def avg_word_length(corpus):
    """Don't forget to docstring me!"""
    '''Returns the average word token length'''
    sum = 0;
    for word in corpus:
       sum += len(word)
    avg = round(sum/len(corpus),2)
    return avg

def max_word_length(corpus):
    '''Returns the longest word token length'''
    #return max(corpus)
    max_list = []
    word_list = []
    for word in corpus:
       max_list.append(len(word))
    maxi = max(max_list)
    for word in corpus:
        if len(word) == maxi:
            word_list.append(word)
    return maxi,word_list

def nr_hapax_words(corpus):
    '''Returns number of hapax words'''
    freqs = {key: 0 for key in corpus}
    hapax_list = []
    for word in corpus:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            hapax_list.append(word)
    return len(hapax_list)


def most_freq_corpus_words(corpus):
    '''Returns most frequent words'''
    sum = 0;
    cnt = Counter(corpus).most_common(10)
    for word in cnt:
       sum += word[1]
    freq_perc = round(sum/len(corpus)*100,2)
    return cnt,freq_perc

def equal_corpus_slices(corpus):
    '''Returns plots of the 10 slices of equal sizes with the percentage of hapaxes'''
    slice_hapax_list=[]
    slice_hapax_list_increased = []
    test =[]
    nr_hapax_each_slice =[]
    nr_hapax_each_sub_slice =[]
    slicesize = round(len(corpus)/10)
    slice_list = [corpus[x:x + slicesize] for x in range(0, len(corpus), slicesize)]
    for slice in slice_list:
        slice_hapax_list.append(round((nr_hapax_words(slice))/len(slice)*100,2))
        nr_hapax_each_slice.append(nr_hapax_words(slice))
        #print("Number of hapax in slice: ",len(haplist),"Percentage: ",round((len(haplist)/len(slice))*100,2) )
    plt.plot(slice_hapax_list)
    plt.title("Question 7.1")
    plt.show()
    for slice_increase in slice_list:
        test.extend(slice_increase)
        slice_hapax_list_increased.append(round((nr_hapax_words(test))/len(test)*100,2))
        nr_hapax_each_sub_slice.append(nr_hapax_words(test))
    plt.plot(slice_hapax_list_increased)
    plt.title("Question 7.2")
    plt.show()
    return slice_hapax_list,slice_hapax_list_increased,nr_hapax_each_slice,nr_hapax_each_sub_slice

def unique_bigrams(corpus):
    '''Returns unique word bigrams and percentage in the corpus'''
    bigrams = list(nltk.bigrams(corpus))
    uni_bigrams = set(bigrams)
    percent_bigram = (len(uni_bigrams)/len(bigrams))*100
    return round(percent_bigram,2)

def unique_trigrams(corpus):
    '''Returns unique word trigrams and percentage in the corpus'''
    trigrams = list(nltk.trigrams(corpus))
    uni_trigrams = set(trigrams)
    percent_trigrams = (len(uni_trigrams)/len(trigrams))*100
    return round(percent_trigrams,2)



def corpus_statistics(corpus):
    """function to take tokenized corpus and prints its stats"""
    print("Assignment 1: WordNet (deadline: 2018-11-27) Name: Kirsten Bassuday")
    print("Part 2:")
    print("=============")
    print("Q1: Number of word tokens and word types in the corpus: ", nr_corpus_words(corpus))
    print("Q2: The average word token length: ", avg_word_length(corpus))
    max, words = max_word_length(corpus)
    print("Q3: The longest word length is: ", max, "The words are: ", words)
    print("Q4.1: The number of hapax words: ", nr_hapax_words(corpus))
    word_tokens, wordtypes = nr_corpus_words(corpus)
    print("Q4.2 Hapax words represent ", round((nr_hapax_words(corpus) / word_tokens * 100), 2), "% ")
    freq_list, freq_perc = most_freq_corpus_words(corpus)
    print("Q5.1: The 10 most frequent words: ", freq_list)
    print("Q5.2: The 10 most frequent words represent: ", freq_perc, "% of the corpus")

    slice_perc, slice_hapax_list_increased,nr_hapax_each_slice,nr_hapax_each_sub_slice = equal_corpus_slices(corpus)
    for j in range(0,len(nr_hapax_each_slice)):
        print("Q6.1: Number of hapax in each slice: ",nr_hapax_each_slice[j], "and its percentage: ", slice_perc[j])
    for i in range(0,len(nr_hapax_each_sub_slice)):
        print("Q6.2: Number of Hapax in each subcorpora: ",nr_hapax_each_sub_slice[i], "and its perecentage:  ", slice_hapax_list_increased[i])

    print("Q8: Percent of unique bigrams: ", unique_bigrams(corpus),"%")
    print("Q9: Percent of unique trigrams: ", unique_trigrams(corpus),"%")

if __name__ == "__main__":
    nr_files = 199
    corpus_text = get_corpus_text(nr_files)
    gold_tokens = get_gold_tokens(nr_files)
    tokens = tokenize_corpus(corpus_text)
    evaluate_tokenization(tokens, gold_tokens)
    corpus_statistics(tokens)
    #slice_perc, slice_hapax_list_increased,nr_hapax_each_slice = equal_corpus_slices(tokens)
    #print(nr_hapax_each_slice)
