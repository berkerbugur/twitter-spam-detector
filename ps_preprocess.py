# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 06:53:34 2018

@author: bugur
"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.stem import SnowballStemmer

### PREPROCESSING THE TWEET FOR ANALYSIS ###
def porter_process(tweet, lower_case = True, stem = True, stop_words = True, gram = 2):
    #When lower_case parameter is present
    if lower_case:
        tweet = tweet.lower()
    words = word_tokenize(tweet)
    words = [w for w in words if len(w) > 2]
    
    #For n-gram model gram is 2 by default. If user enters another value check
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    
    #When stop_words parameter is present
    if stop_words:
        stoppedW = stopwords.words('english')
        words = [word for word in words if word not in stoppedW]
    
    #When the Porter Stemmer parameter stem is present
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

### SNOWBALL PROCESSING, SIMPLE AND QUICKER ###
def snowball_process(tweet):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = [word for word in tweet.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in tweet:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words
