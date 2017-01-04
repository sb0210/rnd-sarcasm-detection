#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
from time import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter
from scipy.stats import randint as sp_randint

import sys
import os

precision_array=[]
recall_array=[]
f_score_array=[]



lang="english"
cross=10
path="data"
results_folder=""
remove_words = ['#sarcasm','video','watch',"@rt"]
remove = "|".join(remove_words)
#training dataset
irony_files=[]
non_irony_files=[]
result_files = []
result_stat_files = []

def get_tweet(line):
    return line.split("\t")[1]

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def apply_svm(test_index,c,C,gamma):
    print "Test file index: "+str(test_index)

    #training tweets
    orig_tweets = []
    clean_tweets =[]
    tweets_feature =[]

    #testing tweets
    test_tweets = []
    orig_test_tweets = []

    #statistical features
    total_sarcastic_tweets = 0
    total_tweets = 0
    sarcastic_tweets_correctly_predicted = 0
    sarcastic_predicted_tweets = 0

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = c)

    print "Reading and cleaning the tweets.."
    train_sarcasm_set = []
    train_nonsarcasm_set = []
    test_sarcasm_set = irony_files[test_index]
    test_nonsarcasm_set = non_irony_files[test_index]
    print "Test Set is ", test_index
    print "Train Set is ",
    for i in xrange(cross):
        if i==test_index:
            continue
        print i,    
        train_sarcasm_set=train_sarcasm_set+irony_files[i]
        train_nonsarcasm_set=train_nonsarcasm_set+non_irony_files[i]
    print ""
    start_time = time()
    for line in train_sarcasm_set:
        clean_tweets.append(get_tweet(line))
        orig_tweets.append(line)
        tweets_feature.append(1)
    for line in train_nonsarcasm_set:
        clean_tweets.append(get_tweet(line))
        orig_tweets.append(line)
        tweets_feature.append(0)
    for line in test_sarcasm_set:
        test_tweets.append(get_tweet(line))
        orig_test_tweets.append(line)
        total_sarcastic_tweets = total_sarcastic_tweets + 1
        total_tweets = total_tweets +1 
    for line in test_nonsarcasm_set:
        test_tweets.append(get_tweet(line))
        orig_test_tweets.append(line)
        total_tweets = total_tweets + 1
    end_time = time()
    print "Cleaning of all tweets done in " + str(end_time - start_time)+" seconds.\n"

    print "Vectorizing the data..."
    start_time = time()
    train_data_features = vectorizer.fit_transform(clean_tweets)
    train_data_features = train_data_features.toarray()
    print train_data_features.shape
    end_time = time()
    print "Vectorization done in " + str(end_time - start_time)+" seconds.\n"


    print "Finding best parameters"
    clf = svm.SVC()
    param_dist = {"gamma": [0.1,0.5,0.7,1.5,2],
                     "C": [0.1,0.2,0.5,0.7,0.9,1.1,1.4,1.8]}

    # n_iter_search = 20
    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
    #                                    n_iter=n_iter_search)

    # start = time()
    # random_search.fit(train_data_features,tweets_feature)
    # grid_scores =  random_search.grid_scores_
    # report(grid_scores)
    # best_score = sorted(grid_scores, key=itemgetter(1), reverse=True)[0]
    # C,gamma = best_score.parameters['C'],best_score.parameters['gamma']
    C,gamma = 1.5,1

    print "Training in progress....."
    print "Training with "+str(len(train_data_features))+" labelled training set."
    start_time = time()
    clf = svm.SVC(C=C,gamma=gamma)
    clf.fit(train_data_features, tweets_feature)
    end_time = time()
    print "Training done in " + str(end_time - start_time)+" seconds.\n"

    model_data=results_folder+"/models"
    if not os.path.exists(model_data):
        os.makedirs(model_data)

    print "Saving the model"
    # joblib.dump(clf, model_data    +'/model'+str(test_index+1)+'.pkl')     
    print "Model Saved"
    # return
    # print "loading the model"
    # clf = joblib.load('data/models/model'+str(test_index+1)+'.pkl')
    # print "model loaded"

    print "Testing in progress"
    start_time = time()
    test_data_features = vectorizer.transform(test_tweets)
    test_data_features = test_data_features.toarray()
    result = clf.predict(test_data_features)
    end_time = time()
    print "Testing in done in " + str(end_time - start_time)+" seconds.\n"

    output_file = result_files[test_index]
    # for res,tw,cl in zip(result,orig_test_tweets,test_tweets):
        # string = str(res).decode("utf-8",'ignore') + "\t"+cl.decode("utf-8",'ignore')+"\t"+ str(tw.split("\t")[1]).decode("utf-8",'ignore')+"\t"+ str(tw.split("\t")[3]).decode("utf-8",'ignore')
        # output_file.write(string)
    print "Output generated in tests/result_file"+str(test_index+1)+".txt file\n"

    for res,tw in zip(result,orig_test_tweets):
        if res ==1:
            sarcastic_predicted_tweets = sarcastic_predicted_tweets + 1
        if res==1 and tw.split("\t")[3][0]=="S":
            sarcastic_tweets_correctly_predicted = sarcastic_tweets_correctly_predicted + 1
        if res==1:
            result_files[i].write("SARCASTIC\t"+tw)
        if res==0:
            result_files[i].write("NON_SARCASTIC\t"+tw)


    result_string="***Test Results***"+"\n"+"Total Training Tweets\t"+str(len(train_data_features))+"\n"+"Total Sarcastic Training Tweets\t"+str(len(train_sarcasm_set))+"\n"+"Total Non-Sarcastic Training Tweets\t"+str(len(train_nonsarcasm_set))+"\n"+"Total Test Tweets\t "+str(len(test_data_features))+"\n"+"Total Sarcastic Tweets present in test\t"+str(len(test_sarcasm_set))+"\n"+"Total Non Sarcastic Tweets present in test\t"+str(len(test_nonsarcasm_set))+"\n"+"Total tweets which were predicted as sarcastic\t"+str(sarcastic_predicted_tweets)+"\n"+"Total tweets which were predicted as non-sarcastic\t"+str(len(test_data_features)-sarcastic_predicted_tweets)+"\n"+"Sarcastic Tweets which were predicted correctly\t"+str(sarcastic_tweets_correctly_predicted)

    try:
        precision = float(sarcastic_tweets_correctly_predicted)/sarcastic_predicted_tweets
        recall = float(sarcastic_tweets_correctly_predicted)/total_sarcastic_tweets
        f_score = float(2*precision*recall)/(precision+recall)
        precision_array.append(precision)
        recall_array.append(recall)
        f_score_array.append(f_score)
        result_string = result_string + "\n\n***Final Results***" +"\n"+ "Precision\t"+str(precision) +"\n"+ "Recall\t\t"+str(recall)+"\n"+"F_score\t\t"+str(f_score)

    except:
        print "\nBad Results."
        result_string = result_string + "\nBad Results"

    print result_string
    result_stat_files[test_index].write(result_string)

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print "Usage: "+sys.argv[0]+ " K path_to_folder language results_folder"
        exit(0)
    cross = int(sys.argv[1])    
    print str(cross)+"-cross fold validation"
    path = sys.argv[2]
    lang=sys.argv[3]
    results_folder = sys.argv[4]
    C=1.3
    gamma=0.1
    try:
        if sys.argv[5]=="test":
            sarcastic_data = path+"/TestSarcasticTweets.txt"
            non_sarcastic_data = path+"/TestNonSarcasticTweets.txt"
    except:
        sarcastic_data = path+"/CleanSarcasticTweets.txt"
        non_sarcastic_data = path+"/CleanNonSarcasticTweets.txt"

    folded_data = results_folder+"/folded_data"
    result_data = results_folder+"/results"
    sarcastic_file = list(open(sarcastic_data,'r'))
    non_sarcastic_file = list(open(non_sarcastic_data,'r'))

    print "Dividing sarcastic tweets file"
    if not os.path.exists(folded_data):
        os.makedirs(folded_data)
    if not os.path.exists(result_data):
        os.makedirs(result_data)
    tot_tweets = len(sarcastic_file)
    tweets_per_file = tot_tweets/cross
    for i in xrange(cross):
        file = open(folded_data+"/sarcasm_set"+str(i+1)+".txt",'w',0)
        for j in xrange(tweets_per_file):
            file.write(sarcastic_file[i*tweets_per_file+j])
    tot_tweets = len(non_sarcastic_file)
    tweets_per_file = tot_tweets/cross
    for i in xrange(cross):
        file = open(folded_data+"/non_sarcasm_set"+str(i+1)+".txt",'w',0)
        for j in xrange(tweets_per_file):
            file.write(non_sarcastic_file[i*tweets_per_file+j])
        file.flush()
        os.fsync(file.fileno())
        file.close()

    for i in xrange(0,cross):
        irony_files.append(list(open(folded_data+"/sarcasm_set"+str(i+1)+".txt",'r')))
        non_irony_files.append(list(open(folded_data+"/non_sarcasm_set"+str(i+1)+".txt",'r')))
        result_files.append(open(result_data+"/result_file"+str(i+1)+".txt",'w'))
        result_stat_files.append(open(result_data+"/result_stat_file"+str(i+1)+".txt",'w'))
    
    print "Diving done"

    print "Applying "+str(cross)+"-cross fold validation\n"


    for i in xrange(0,cross):
        apply_svm(i,3000,C,gamma)
        print "\n"
        # break

    print "Overall result"
    try:
        precision = sum(precision_array)/float(len(precision_array))
        recall = sum(recall_array)/float(len(recall_array))
        f_score = (2*precision*recall)/(precision + recall)
        result_string = "\n\n***Final Results***" +"\n"+ "Precision\t"+str(precision) +"\n"+ "Recall\t\t"+str(recall)+"\n"+"F_score\t\t"+str(f_score)
        print result_string
    except:
        print "Bad result"
        pass
