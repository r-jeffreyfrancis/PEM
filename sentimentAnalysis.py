# -*- coding: utf-8 -*-
# from app import keyword
"""
Spyder Editor

This is a temporary script file.
"""

import twitter
import csv
import pickle
import sys
import imp



testDataSet = ""
search_term = ""

# initialize api instance
twitter_api = twitter.Api(consumer_key='CkFtH6uSH8WAtK2ZXzu2xOD1t',
                        consumer_secret='TeXQcNzlHRfrWDOWnZ0wFZor6sl6r729erMpe4Is57ttG6GMEZ',
                        access_token_key='853763464015233024-U0skgfqIXP7O3YeZ056NdfkNv5Ssw7V',
                        access_token_secret='vIlR4q3uyxA4dtISRPFBs8MO3xuXGAPfuXyUVSNpRygNs')

# test authentication
#print(twitter_api.VerifyCredentials())

train_or_predict = 0

# ------------------------------------------------------------------------

def buildTestSet(data, new):
    
    if (new ==1):
        search_keyword = data
        try:
            tweets_fetched = twitter_api.GetSearch(search_keyword, count=100)
            
            print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
    
            return [{"text":status.text, "label":None} for status in tweets_fetched]
        except:
            print("Unfortunately, something went wrong..")
            return None
    else:
        testSet=[]
        with open(data,'rt') as csvfile:
            lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
            for row in lineReader:
                testSet.append({"tweet_id":row[0], "label":row[1], "text":row[2]})
        return testSet
# ------------------------------------------------------------------------

search_term = input("Enter a search keyword: ")
"""
MAIN
"""
# def main(arg):
# if __name__ == '__main__':
#     print("Passed in search term")
#     search_term = sys.argv[1]
#     print(search_term)
#     main(search_term)
    
testDataSet = buildTestSet(search_term,1)
    # print(testDataSet[0:4])

# def main(search_term):
#     import app
#     search_term = app.keyword
#     print(search_term)
#     buildTestSet(search_term,1)
#labledTestData = '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/twitter-sentiment-analysis2/train.csv'
#testDataSet = buildTestSet(labledTestData, 0)



# ------------------------------------------------------------------------

def buildTrainingSet(tweetDataFile):
   
    #import time 

    trainingDataSet=[]
    
    with open(tweetDataFile,'rt') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            #trainingDataSet.append({"tweet_id":row[2], "label":row[1], "topic":row[0], "text":row[4]})
            #trainingDataSet.append({"tweet_id":row[1], "label":row[0], "text":row[4]})
            trainingDataSet.append({"label":row[0], "text":row[5]})
    
    return trainingDataSet

# ------------------------------------------------------------------------

if train_or_predict == 0:
    #corpusFile = "YOUR_FILE_PATH/corpus.csv"
    tweetDataFile = '/Users/r.jeffreyfrancis@ibm.com/Downloads/twitter_corpus-master/full-corpus.csv'
    tweetDataFile2= '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/training.1600000.edited.csv'
    tweetDataFile3_vb = '/Users/jenny.pan@ibm.com/Documents/GitHub/PEM/sentiment140_lite.csv'
    trainingData = buildTrainingSet(tweetDataFile3_vb)

# ------------------------------------------------------------------------

import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        singleProccessedTweet=[]
        pureTweetData=[]
        processedTweets=[]
        for tweet in list_of_tweets:
            singleProccessedTweet=self._processTweet(tweet["text"])
            processedTweets.append((singleProccessedTweet,tweet["label"]))
            pureTweetData.append(singleProccessedTweet)
        return processedTweets, pureTweetData
    
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor = PreProcessTweets()
preprocessedTestSet, pureTestData = tweetProcessor.processTweets(testDataSet)

if train_or_predict==0:
    preprocessedTrainingSet, pureTrainData = tweetProcessor.processTweets(trainingData)


# ------------------------------------------------------------------------

import nltk 

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

# ------------------------------------------------------------------------

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

# ------------------------------------------------------------------------
if train_or_predict==0:
    # Now we can extract the features and train the classifier 
    word_features = buildVocabulary(preprocessedTrainingSet)
    trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)

# ------------------------------------------------------------------------

    NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

# -Load Pre-Trained Model-----------------------------------------------------------------------
else:
    pretrainedClassifier='/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/From VB/N1000_model.pickle'
    f = open(pretrainedClassifier, 'rb')
    NBayesClassifier = pickle.load(f)
    f.close()

# ------------------------------------------------------------------------
#we need to save the data from the extract features step
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

# ------------------------------------------------------------------------

# get the majority vote
if NBResultLabels.count('4') > NBResultLabels.count('0'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('4')/len(NBResultLabels)) + "%")
else: 
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('0')/len(NBResultLabels)) + "%")
    
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
	
import pyLDAvis.gensim
	
def topicModeling (corpus, dictionary):
    
    ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

#    x = ldamodel.show_topics() #show generated topics 
   
    # pyLDAvis.enable_notebook()
  
    topicModel = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    
    pyLDAvis.save_html(topicModel, 'templates/visualization.html')
    pyLDAvis.show(topicModel)

#create dictionary
modDict = gensim.corpora.Dictionary(pureTestData)
#create corpus:  bags of words, tuples representing frequency
corp = [modDict.doc2bow(genTestSet) for genTestSet in pureTestData]

topicModeling(corp, modDict)
