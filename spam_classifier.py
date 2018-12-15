# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 05:25:10 2018

@author: bugur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import ps_preprocess as pp

#Read dataset
tweets = pd.read_csv('C:\\Users\\bugur\\OneDrive\\Desktop\\TwitterSpamDetector\\Datasets\\SpamAndHam.csv')

#Showcase dataframe
tweet_preview = tweets.tail(10)
tweet_group = tweets.groupby('Label').describe()

print('*********** DATASET ***********')
print(tweet_preview)
print('\n------------------\n')
print('*********** DATASET GROUPED BY LABEL ***********')
print(tweet_group)
print('\n------------------\n')


#Get number of tweets from the matris tweets and create offsets for training and testing purposes
totalTweets = tweets['TweetText'].shape[0]
trainOffset, testOffset = list(), list()

#Using random selection for separation with sensitivity of 0.8
#Meaning that 80% of the dataset will be use for training purposes
#Remaining 20% will be used for testing and discovering accuracy 
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.8:
        trainOffset += [i]
    else:
        testOffset += [i]

#Loading the training and testing data with the randomly selected location offsets
train_data = tweets.loc[trainOffset]
test_data = tweets.loc[testOffset]

#Showcase training data
train_data.reset_index(inplace = True)
train_data.drop(['index'], axis = 1, inplace = True)
train_preview = train_data.head(10)
print('*********** SHOWCASE OF TRAINING DATA ***********')
print(train_preview)
print('\n')

#Training data grouping by label
train_group = train_data['Label'].value_counts()
print('*********** TRAINING DATA GROUPED BY LABEL ***********')
print(train_group)
print('\n------------------\n')

spam_words = ' '.join(list(tweets[tweets['Label'] == 1]['TweetText']))
spam_Vcloud = WordCloud(width = 512, height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_Vcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** SPAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')

ham_words = ' '.join(list(tweets[tweets['Label'] == 0]['TweetText']))
ham_Vcloud = WordCloud(width = 512, height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_Vcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
print('*********** HAM WORD CLOUD ***********')
plt.show()
print('\n------------------\n')



### BAYESIAN CLASSIFICATION METHOD ###
class BayesianClassifier(object):
    def __init__(self, train_data, method = 'tf-idf'):
        self.tweets, self.label = train_data['TweetText'], train_data['Label']
        self.method = method
    def train(self):
        self.TF_and_IDF()
        if self.method == 'tf-idf':
            self.TF_IDF
        else:
            self.bayesian_prob()
    def bayesian_prob(self):
        self.is_spam = dict()
        self.is_ham = dict()
        for word in self.tf_spam:
            self.is_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.is_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + len(list(self.tf_ham.keys())))
        self.spam_prob, self.ham_prob = self.spam_tweets / self.total_tweets, self.ham_tweets / self.total_tweets
    def TF_and_IDF(self):
        tweet_count = self.tweets.shape[0]
        self.spam_tweets, self.ham_tweets = self.label.value_counts()[1], self.label.value_counts()[0]
        self.total_tweets = self.spam_tweets + self.ham_tweets
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(tweet_count):
            tweet_processed = pp.porter_process(self.tweets[i])
            count = list() #For IDF to keep track of whether the word has occured in tweet text or not
            for word in tweet_processed:
                if self.label[i]:
                    self.tf_spam[word] = self.tf_ham.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.label[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1
    def TF_IDF(self):
        self.is_spam = dict()
        self.is_ham = dict()
        self.total_tfidf_spam = 0
        self.total_tfidf_ham = 0
        for word in self.tf_spam:
            self.is_spam[word] = (self.tf_spam[word]) * log((self.spam_tweets + self.ham_tweets) / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.total_tfidf_spam += self.spam_prob[word]
        for word in self.tf_spam:
            self.is_spam[word] =(self.is_spam[word] + 1) / (self.total_tfidf_spam + len(list(self.is_spam.keys())))
        for word in self.tf_ham:
            self.is_ham[word] = (self.is_ham[word]) * log((self.spam_tweets + self.ham_tweets) / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.total_tfidf_ham += self.is_ham[word]
        for word in self.tf_ham:
            self.is_ham[word] = (self.is_ham[word] + 1) /(self.total_tfidf_ham + len(list(self.is_ham.keys())))
        self.spam_prob, self.ham_prob = self.spam_tweets / self.total_tweets, self.ham_tweets / self.total_tweets
    def classify(self, processed_tweet):
        pSpam, pHam = 0, 0
        for word in processed_tweet:
            if word in self.is_spam:
                pSpam += log(self.is_spam[word])
            else:
                if self.method == 'tf-idf':
                    pSpam -= log(self.total_tfidf_spam + len(list(self.is_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + len(list(self.is_spam.keys())))
            if word in self.is_ham:
                pHam += log(self.is_ham[word])
            else:
                if self.method == 'tf-idf':
                    pHam -= log(self.total_tfidf_ham + len(list(self.is_ham.keys())))
                else:
                    pHam -= log(self.ham_words + len(list(self.is_ham.keys())))
            pSpam += log(self.spam_prob)
            pHam += log(self.ham_prob)
        return pSpam >= pHam
    def predict(self, test_data):
        result = dict()
        for (i, tweet) in enumerate(test_data):
            processed_tweet = pp.porter_process(tweet)
            result[i] = int(self.classify(processed_tweet))
        return result
def metrics(label, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(label)):
        true_pos += int(label[i] == 1 and predictions[i] == 1)
        true_neg += int(label[i] == 0 and predictions[i] == 0)
        false_pos += int(label[i] == 0 and predictions[i] == 1)
        false_neg += int(label[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F-Score: ', Fscore)
    print('Accuracy: ', accuracy)
    print('\n------------------\n')

Spam_Classify_tfidf = BayesianClassifier(train_data, 'tf-idf')
Spam_Classify_tfidf.train()
preds_tfidf = Spam_Classify_tfidf.predict(test_data['TweetText'])
metrics(test_data['Label'], preds_tfidf)

Spam_Classify_BoW = BayesianClassifier(train_data, 'bow')
Spam_Classify_BoW.train()
preds_BoW = Spam_Classify_BoW.predict(test_data['TweetText'])
metrics(test_data['Label'], preds_BoW)
