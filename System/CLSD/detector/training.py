#imports 
import re
import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.externals import joblib
import os
import sys
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter
from scipy.stats import randint as sp_randint
from textblob import TextBlob
from param import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test=""
def get_vocabulary_size(sets):
    s = []
    for s1 in sets:
        s=s+s1.split(" ")
    return len(set(s))

def get_total_words(sets):
    return sum([len(s.split(" ")) for s in sets ])

def get_average_tweet_length(tweet_set):
    total_length=0
    for tw in tweet_set:
        total_length=total_length+len(tw.split(" "))
    return float(total_length)/len(tweet_set)

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def training(folder,model,C,gamma,factor=20):
    irony_file = list(open(folder+"/"+test+"SarcasticTweets.txt",'r'))+list(open(folder+"/"+test+"NonSarcasticTweets.txt",'r'))    

    train_sarcasm_set = []
    train_nonsarcasm_set = []
    print "Train Set is ",file
    for line in irony_file:
        if line.split("\t")[3][0]=="S":
            train_sarcasm_set.append(line)
        else:
            train_nonsarcasm_set.append(line)        
    clean_tweets=[]
    tweets_feature=[]

    start_time = time.time()
    for line in train_sarcasm_set:
        clean_tweets.append(get_clean_tweet(line))
        tweets_feature.append(1)
    for line in train_nonsarcasm_set:
        clean_tweets.append(get_clean_tweet(line))
        tweets_feature.append(0)
    end_time = time.time()

    print "Dataset Features"
    skew =  float(len(train_sarcasm_set))/(len(train_nonsarcasm_set)+len(train_nonsarcasm_set))
    avg_tweet_length = get_average_tweet_length(train_nonsarcasm_set+train_sarcasm_set)
    vocabulary_size = get_vocabulary_size(train_nonsarcasm_set+train_sarcasm_set)
    total_size = get_total_words(train_nonsarcasm_set+train_sarcasm_set)
    avg_word_repetition = float(total_size)/vocabulary_size
    print "Sarcastic Tweets in training ",len(train_sarcasm_set)
    print "Non Sarcastic Tweets in training ",len(train_nonsarcasm_set)
    print "Skew ",skew*100," % Sarcastic Comments",(1-skew)*100," % Non Sarcastic Comments"
    print "Total Number of words", total_size
    print "vocabulary_size",vocabulary_size
    print "Average Tweet length",avg_tweet_length
    print "Word Repetition Average",avg_word_repetition

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 ngram_range = (1,2), \
                                 lowercase = True, \
                                 max_features = vocabulary_size/factor)


    print "Vectorizing the data..."
    start_time = time.time()
    train_data_features = vectorizer.fit_transform(clean_tweets)
    print vectorizer.get_feature_names()
    train_data_features = train_data_features.toarray()
    print train_data_features.shape
    print train_data_features

    i=0
    for tweet in clean_tweets:
        if par :
            continue
        contra_feature2 = 0
        contra_feature3 = 0
        positivity = 0
        negativity = 0

        first_half_positivity3 = 0
        second_half_positivity3 = 0
        first_half_negativity3 = 0
        second_half_negativity3 = 0
        third_half_positivity3 = 0
        third_half_negativity3 = 0

        first_half_positivity2 = 0
        second_half_positivity2 = 0
        first_half_negativity2 = 0
        second_half_negativity2 = 0
        blob = TextBlob(tweet)
 
        length = len(tweet.split(" "))
        if length<3:
            train_data_features[i][-15:] = [positivity,negativity,contra_feature2,contra_feature3,first_half_positivity2,first_half_negativity2,second_half_positivity2,second_half_negativity2,first_half_positivity3,first_half_negativity3,second_half_positivity3,second_half_negativity3,third_half_positivity3,third_half_negativity3,length]
            i=i+1
            continue
        part1 = tweet.split(" ")[0:length/3]    
        part2 = tweet.split(" ")[length/3:2*length/3]    
        part3 = tweet.split(" ")[2*length/3:]    

        blobs = []
        blobs.append(TextBlob(" ".join(part1)))
        blobs.append(TextBlob(" ".join(part2)))
        blobs.append(TextBlob(" ".join(part3)))

        try:
            polarities = []
            for b in blobs:
                polarities.append(b.sentiment.polarity)

            if (polarities[0]*polarities[1]<0 or polarities[0]*polarities[2]<0 or polarities[2]*polarities[1]<0):
                contra_feature3 = 1 

            first_half_positivity3 = (polarities[0]>0)      
            first_half_negativity3 = (polarities[0]<0)      
            second_half_positivity3 = (polarities[1]>0)      
            second_half_negativity3 = (polarities[1]<0)      
            third_half_positivity3 = (polarities[2]>0)      
            third_half_negativity3 = (polarities[2]<0)      
        except:
            pass

        part1 = tweet.split(" ")[0:length/2]    
        part2 = tweet.split(" ")[length/2:]    

        blobs = []
        blobs.append(TextBlob(" ".join(part1)))
        blobs.append(TextBlob(" ".join(part2)))

        try :
            polarities = []
            for b in blobs:
                polarities.append(b.sentiment.polarity)

            if (polarities[0]*polarities[1]<0):
                contra_feature2 = 1    


            first_half_positivity2 = (polarities[0]>0)      
            first_half_negativity2 = (polarities[0]<0)      
            second_half_positivity2 = (polarities[1]>0)      
            second_half_negativity2 = (polarities[1]<0)      
        except:
            pass    
        try:    
            if blob.sentiment.polarity > 0:
                positivity = 1
            if blob.sentiment.polarity < 0:
                negativity = 1
        except:
            print tweet
        train_data_features[i][-15:] = [positivity,negativity,contra_feature2,contra_feature3,first_half_positivity2,first_half_negativity2,second_half_positivity2,second_half_negativity2,first_half_positivity3,first_half_negativity3,second_half_positivity3,second_half_negativity3,third_half_positivity3,third_half_negativity3,length]

        i=i+1
    end_time = time.time()

    v_name = model.split("/")
    vectorizer_name = ""
    for i in xrange(len(v_name)-1):
        vectorizer_name = vectorizer_name+v_name[i]+"/"
    vectorizer_name = vectorizer_name+"vectorizer_"+v_name[len(v_name)-1]
    joblib.dump(vectorizer,vectorizer_name)
    print "Vectorization done and stored in " + str(end_time - start_time)+" seconds.\n"


    print "Finding best parameters"
    clf = svm.SVC()
    # param_dist = {"gamma": [0.5],#1,2,3,5,10],
    #                  "C": [0.1,0.2,0.5,0.7,0.9,1.1,1.4,1.8,3,4,7,5,6,8,10]}

    # n_iter_search = 15
    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
    #                                    n_iter=n_iter_search)

    # random_search.fit(train_data_features,tweets_feature)
    # grid_scores =  random_search.grid_scores_
    # report(grid_scores)
    # best_score = sorted(grid_scores, key=itemgetter(1), reverse=True)[0]
    # C,gamma = best_score.parameters['C'],best_score.parameters['gamma']
    
    #for english italian
    # if folder=="en-it":
    #     C,gamma = 1,0.3

    #for czech italian
    # else:
        # C,gamma = 1,0.3
    # if folder == "cs-cs":
    
    # C,gamma =  1.6,0.5    for czech language BD
    # C,gamma =  1.6,0.5    #for czech language BD
    #C,gamma = 1,0.3 for it
    C,gamma = 1,0.3

    print "Training in progress....."
    print "Training with "+str(len(train_data_features))+" labelled training set."
    start_time = time.time()
    clf = svm.SVC(C=C,gamma=gamma)    
    clf.fit(train_data_features, tweets_feature)
    end_time = time.time()
    print "Training done in " + str(end_time - start_time)+" seconds.\n"
    print "Saving the model"
    joblib.dump(clf, model)     
    print "Model Saved"

if __name__ == "__main__":
    if len(sys.argv)<6:
        print "folder, model, max_features, test, num, lang"
        exit(0)
    folder=sys.argv[1]#raw_input("Enter name of training Folder : ")
    model=sys.argv[2]#raw_input("Enter name of model file to be saved : ")
    factor=int(sys.argv[3])#input("Max number of features : ")
    t=sys.argv[4]#input("Test or not??")
    num = sys.argv[5]
    lang = sys.argv[6]
    C,gamma = 1.5,1
    # C= input("C : ")
    # gamma = input("Gamma : ")
    if not os.path.exists(os.path.dirname(model)):
        os.makedirs(os.path.dirname(model))
    if t=="test":
        test="Test"
        os.system("bash ./test.sh "+num+" " + folder)
    training(folder,model,C,gamma,factor)

