# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import twitter
import csv
import pickle
import pandas as pd
import pyLDAvis

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# initialize api instance
twitter_api = twitter.Api(consumer_key='CkFtH6uSH8WAtK2ZXzu2xOD1t',
                        consumer_secret='TeXQcNzlHRfrWDOWnZ0wFZor6sl6r729erMpe4Is57ttG6GMEZ',
                        access_token_key='853763464015233024-U0skgfqIXP7O3YeZ056NdfkNv5Ssw7V',
                        access_token_secret='vIlR4q3uyxA4dtISRPFBs8MO3xuXGAPfuXyUVSNpRygNs')

# test authentication
#print(twitter_api.VerifyCredentials())

#if train = 0 it will run pretrained model
train = 1

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
                #testSet.append({"label":row[0], "text":row[5]})
                testSet.append({"label":None, "text":row[3]})
        return testSet
# ------------------------------------------------------------------------

#search_term = input("Enter a search keyword: ")
#testDataSet = buildTestSet(search_term,1)
#labledTestData = '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/From VB/test_positive.csv'
labledTestData ='/Users/r.jeffreyfrancis@ibm.com/Downloads/replies (1).csv'
testDataSet = buildTestSet(labledTestData, 0)

#print(testDataSet[0:4])

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

if train == 1:
    #corpusFile = "YOUR_FILE_PATH/corpus.csv"
    tweetDataFile = '/Users/r.jeffreyfrancis@ibm.com/Downloads/twitter_corpus-master/full-corpus.csv'
    tweetDataFile2= '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/training.1600000.edited.csv'
    tweetDataFile3_vb = '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/From VB/sentiment140_lite.csv'
    trainingData = buildTrainingSet(tweetDataFile3_vb)

# ------------------------------------------------------------------------

import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 

class PreProcessTweets:
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.stopwords.append('rt')
        self._stopwords = set(self.stopwords + list(punctuation) + ['AT_USER','URL'])
        self.semiProcessed=[]
        
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
        self.semiProcessed.append(tweet)
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessorTest = PreProcessTweets()
preprocessedTestSet, pureTestData = tweetProcessorTest.processTweets(testDataSet)
semiProcessedTestData = tweetProcessorTest.semiProcessed

if train==1:
    tweetProcessorTrain = PreProcessTweets()
    preprocessedTrainingSet, pureTrainData = tweetProcessorTrain.processTweets(trainingData)


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
if train==1:
    # Now we can extract the features and train the classifier 
    word_features = buildVocabulary(preprocessedTrainingSet)
    trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)

# ------------------------------------------------------------------------

    NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

# -Load Pre-Trained Model-----------------------------------------------------------------------
else:
    pretrainedClassifier='/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/From VB/N20000_model.pickle'
    f = open(pretrainedClassifier, 'rb')
    NBayesClassifier = pickle.load(f)
    f.close()
    
    extractedFeatures='/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/DATA/From VB/N20000_word_features.pickle'
    f = open(extractedFeatures, 'rb')
    extract_features = pickle.load(f)
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
	
def topicModeling (corpus, dictionary, texts):
    
    ldamodel = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=5)
    
    x = ldamodel.show_topics() #show generated topics 

    #----------------------------------------------------------
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
    #-------Generate Visualization------------------------------
    
    pyLDAvis.enable_notebook()
  
    topicModel = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    
    pyLDAvis.save_html(topicModel, '/Users/r.jeffreyfrancis@ibm.com/Documents/projects/PEM/elon.html')
    
    pyLDAvis.show(topicModel)
    
    return x, sent_topics_df
'''
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
'''
#create dictionary
modDict = gensim.corpora.Dictionary(pureTestData)
#create corpus:  bags of words, tuples representing frequency
corp = [modDict.doc2bow(genTestSet) for genTestSet in pureTestData]

tops, df_topic_sents_keywords = topicModeling(corp, modDict, semiProcessedTestData)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess

def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable
def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
    return sentenceValue
def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))
    return average
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary
positiveText=[]
negativeText=[];
for i in range(int(max(df_dominant_topic["Dominant_Topic"]))+1):
    positiveText.append("")
    negativeText.append("")
for i in range(len(df_dominant_topic["Dominant_Topic"])):
        domtopic=int(df_dominant_topic["Dominant_Topic"][i])
        if(NBResultLabels[i]=='4'):
            positiveText[domtopic]=positiveText[domtopic]+semiProcessedTestData[i]
        if(NBResultLabels[i]=='0'):
            negativeText[domtopic]=negativeText[domtopic]+semiProcessedTestData[i]
positiveFreqTable=[]
negativeFreqTable=[]
positiveSentences=[]
negativeSentences=[]
positiveSentenceScores=[]
negativeSentenceScores=[]
positiveThreshold=[]
negativeThreshold=[]
positiveSummary=[]
negativeSummary=[]
for i in range(len(positiveText)):
    #domtopic=int(df_dominant_topic["Dominant_Topic"][i])
    positiveFreqTable.append(_create_frequency_table(positiveText[i]))
    negativeFreqTable.append(_create_frequency_table(negativeText[i]))
for i in range(len(positiveText)):   
    positiveSentences.append(sent_tokenize(positiveText[i]))
    negativeSentences.append(sent_tokenize(negativeText[i]))
for i in range(len(positiveText)):    
    positiveSentenceScores.append(_score_sentences(positiveSentences[i], positiveFreqTable[i]))
    negativeSentenceScores.append(_score_sentences(negativeSentences[i], negativeFreqTable[i]))
for i in range(len(positiveText)):     
    positiveThreshold.append(_find_average_score(positiveSentenceScores[i]))
    negativeThreshold.append(_find_average_score(negativeSentenceScores[i]))
for i in range(len(positiveText)):     
    positiveSummary.append(_generate_summary(positiveSentences[i], positiveSentenceScores[i], 1.5 * positiveThreshold[i]))
    negativeSummary.append(_generate_summary(negativeSentences[i], negativeSentenceScores[i], 1.5 * negativeThreshold[i]))
for i in range(len(positiveText)): 
    print("The following is the positive summary for topic "+str(i)+":"+ positiveSummary[i])
    print("********************************")
    print("The following is the negative summary for topic "+str(i)+":"+ negativeSummary[i])
    print("********************************")
