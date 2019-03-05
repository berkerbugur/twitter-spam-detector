# -*- coding: utf-8 -*-

import pickle as pk

def saveMNB(vectorizer, classifier):
    '''
    save classifier to disk
    '''
    with open('modelMNB.pkl', 'wb') as infile:
        pk.dump((vectorizer, classifier), infile)
        
def loadMNB():
    '''
    load classifier from disk
    '''
    with open('modelMNB.pkl', 'rb') as file:
      vectorizer, classifier = pk.load(file)
    return vectorizer, classifier