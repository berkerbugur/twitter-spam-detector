# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 05:25:10 2018

@author: bugur
"""

import pickle as pk
import pandas as pd
import numpy as np
import tweet_catch as tC
import matplotlib.pyplot as plt
import ps_preprocess as pp
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score

#Read dataset
tweets = pd.read_csv('C:\\Users\\bugur\\OneDrive\\Desktop\\TwitterSpamDetector\\Datasets\\SpamAndHam.csv')
tweets['length'] = tweets['TweetText'].apply(len)

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
train_preview = train_data.head(15)
print('*********** SHOWCASE OF TRAINING DATA ***********')
print(train_preview)
print('\n')

#Training data grouping by label
train_group = train_data['Label'].value_counts()
print('*********** TRAINING DATA GROUPED BY LABEL ***********')
print(train_group)
print('\n------------------\n')

tweetFeatures = tweets['TweetText'].copy()
tweetFeatures = tweetFeatures.apply(pp.snowball_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(tweetFeatures)

features_train, features_test, labels_train, labels_test = train_test_split(features, tweets['Label'], test_size=0.3, random_state=111)

def save(vectorizer, classifier):
    '''
    save classifier to disk
    '''
    with open('model.pkl', 'wb') as infile:
        pk.dump((vectorizer, classifier), infile)
        
def load():
    '''
    load classifier from disk
    '''
    with open('model.pkl', 'rb') as file:
      vectorizer, classifier = pk.load(file)
    return vectorizer, classifier

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
print('MULTINOMIAL NAIVE BAYES SCORE:', accuracy_score(labels_test,prediction))

save(vectorizer, mnb)

print("\nMNB Classifier accuracy {:.2f}%".format(mnb.score(features_test, labels_test) * 100))

prediction = mnb.predict(features_test)
fscore = metrics.f1_score(labels_test, prediction, average='macro')
print("\nMNB F1 score is: {:.2f}".format(fscore))

vectorizer, mnb = load()
print("*********** TEST PHASE WITH LIVE TWEET ***********\n")



twitter_client = tC.TwitterClient()
tweets = twitter_client.get_live_feed(1)
tweetList = []
tweetList.append(tweets[0].text)

input_transformed = vectorizer.transform(tweetList)
prediction = mnb.predict(input_transformed)

print('Analyzed live-tweet:', tweetList)
print('\nAccording to MNB Classification this tweet is', 'SPAM' if prediction else 'HAM')