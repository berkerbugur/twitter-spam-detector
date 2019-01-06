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
import seaborn as sns
import ps_preprocess as pp
import warnings
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

#Read dataset
tweets = pd.read_csv('SpamAndHam.csv')
tweets['length'] = tweets['TweetText'].apply(len)

#Showcase data
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

features_train, features_test, labels_train, labels_test = train_test_split(features, tweets['Label'], test_size=0.2, random_state=111)

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

alg_name = []
alg_accuracies = []
alg_f1 = []

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
conf_mat = metrics.confusion_matrix(labels_test, prediction)
print('MULTINOMIAL NAIVE BAYES SCORE:', accuracy_score(labels_test, prediction))
alg_name.append('Multinomial Naive Bayes')
alg_accuracies.append(mnb.score(features_test, labels_test) * 100)

bnb = BernoulliNB()
bnb.fit(features_train, labels_train)
pred_bnb = bnb.predict(features_test)
conf_mat_bnb = metrics.confusion_matrix(labels_test, pred_bnb)
print('BERNOULLI NAIVE BAYES SCORE:', accuracy_score(labels_test,pred_bnb))
alg_name.append('Bernoulli Naive Bayes')
alg_accuracies.append(bnb.score(features_test, labels_test) * 100)

svc = SVC(gamma="scale")
svc.fit(features_train, labels_train)
pred_svc = svc.predict(features_test)
conf_mat_svc = metrics.confusion_matrix(labels_test, pred_svc)
print('SUPPORT VECTOR MACHINE SCORE:', accuracy_score(labels_test, pred_svc))
alg_name.append('Support Vector Machine')
alg_accuracies.append(svc.score(features_test, labels_test) * 100)

saveMNB(vectorizer, mnb)

print("\nMNB Classifier accuracy {:.2f}%".format(mnb.score(features_test, labels_test) * 100))
print("BNB Classifier accuracy {:.2f}%".format(bnb.score(features_test, labels_test) * 100))
print("SVC Classifier accuracy {:.2f}%".format(svc.score(features_test, labels_test) * 100))



mnb_fscore = metrics.f1_score(labels_test, prediction, average='macro')
print("\nMNB F1 score is: {:.2f}".format(mnb_fscore))
alg_f1.append(mnb_fscore)

bnb_fscore = metrics.f1_score(labels_test, pred_bnb, average='macro')
print("BNB F1 score is: {:.2f}".format(bnb_fscore))
alg_f1.append(bnb_fscore)

svc_fscore = metrics.f1_score(labels_test, pred_svc, average='weighted')
print("SVC F1 score is: {:.2f}".format(svc_fscore))
alg_f1.append(svc_fscore)

data = pd.DataFrame({'alg_name': alg_name, 'alg_accuracies': alg_accuracies})
sorted_data = data.reindex((data['alg_accuracies'].sort_values(ascending=False)).index.values)

plt.subplots(figsize=(10,6))
sns.barplot(x=sorted_data['alg_name'], y=sorted_data['alg_accuracies'], edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=-45)
plt.xlabel('Algorithm Name')
plt.ylabel('Algorithm Accuracy')
plt.title('Algorithm Train Accuracy Comparison')
plt.show()

dataF1 = pd.DataFrame({'alg_name': alg_name, 'alg_f1': alg_f1})
sorted_dataF1 = dataF1.reindex((dataF1['alg_f1'].sort_values(ascending=False)).index.values)

plt.subplots(figsize=(10,6))
sns.barplot(x=sorted_dataF1['alg_name'], y=sorted_dataF1['alg_f1'], edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=-45)
plt.xlabel('Algorithm Name')
plt.ylabel('Algorithm F1 Score')
plt.title('Algorithm F1 Accuracy Score Comparison')
plt.show()


#Confusion Matrix For MNB
labels = ['HAM', 'SPAM']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat)
plt.title('Confusion Matrix Of The MNB Classifier\n')
fig.colorbar(cax)
tick_marks = np.arange(len(labels))
plt.xlabel('\nPredicted Label')
plt.ylabel('True Label')
plt.xticks(tick_marks, labels, rotation=0)
plt.yticks(tick_marks, labels)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat[i][j]))
plt.show()

vectorizer, mnb = loadMNB()


'''
print("\n*********** TEST PHASE WITH LIVE TWEET ***********\n")

twitter_client = tC.TwitterClient()
tweets = twitter_client.get_live_feed(1)
tweetList = []
tweetList.append(tweets[0].text)
accused_user = tweets[0].user.screen_name

input_transformed = vectorizer.transform(tweetList)
prediction = mnb.predict(input_transformed)

print('Analyzed live-tweet:', tweetList)
print('By user: @'+ accused_user)
print('\nAccording to MNB Classification this tweet is', 'SPAM' if prediction else 'HAM')

'''
