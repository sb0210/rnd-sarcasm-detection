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
import sys
import os
from textblob import TextBlob
from param import *
test=""



def testing(folder,model,output_file):
    test_file = list(open(folder+"/"+test+"SarcasticTweets.txt",'r'))+list(open(folder+"/"+test+"NonSarcasticTweets.txt",'r'))    

    
    #statistical features
    total_sarcastic_tweets = 0
    total_tweets = 0
    sarcastic_tweets_correctly_predicted = 0
    sarcastic_predicted_tweets = 0

    v_name= model.split("/")

    vectorizer_name = ""
    for i in xrange(len(v_name)-1):
        vectorizer_name = vectorizer_name+v_name[i]+"/"
    vectorizer_name = vectorizer_name+"vectorizer_"+v_name[len(v_name)-1]    
    vectorizer = joblib.load(vectorizer_name)

    print "Reading and cleaning the tweets.."
    test_sarcasm_set = []
    test_nonsarcasm_set = []
    orig_test_tweets = []
    tweets_feature=[]
    test_tweets=[]
    print "Test Set is ", folder
    for line in test_file:
        try:
            if line.split("\t")[3][0]=="S":
                test_sarcasm_set.append(line)
            else:
                test_nonsarcasm_set.append(line)
        except:
            print line
            return        
    start_time = time.time()
    for line in test_sarcasm_set:
        test_tweets.append(get_clean_tweet(line))
        orig_test_tweets.append(line)
        total_sarcastic_tweets = total_sarcastic_tweets + 1
        total_tweets = total_tweets +1 
    for line in test_nonsarcasm_set:
        test_tweets.append(get_clean_tweet(line))
        orig_test_tweets.append(line)
        total_tweets = total_tweets + 1
    end_time = time.time()
    print "Cleaning of all tweets done in " + str(end_time - start_time)+" seconds.\n"

    print "Loading the model"
    start_time = time.time()
    clf = joblib.load(model)
    end_time = time.time()
    print "Model Loaded in " + str(end_time - start_time)+" seconds.\n"

    print "Testing in progress"
    start_time = time.time()
    test_data_features = vectorizer.fit_transform(test_tweets)
    test_data_features = test_data_features.toarray()

    i=0
    for tweet in test_tweets:
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
            test_data_features[i][-15:] = [positivity,negativity,contra_feature2,contra_feature3,first_half_positivity2,first_half_negativity2,second_half_positivity2,second_half_negativity2,first_half_positivity3,first_half_negativity3,second_half_positivity3,second_half_negativity3,third_half_positivity3,third_half_negativity3,length]
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
        test_data_features[i][-15:] = [positivity,negativity,contra_feature2,contra_feature3,first_half_positivity2,first_half_negativity2,second_half_positivity2,second_half_negativity2,first_half_positivity3,first_half_negativity3,second_half_positivity3,second_half_negativity3,third_half_positivity3,third_half_negativity3,length]
    
        i=i+1
    end_time = time.time()

    result = clf.predict(test_data_features)
    end_time = time.time()
    print "Testing in done in " + str(end_time - start_time)+" seconds.\n"

    output=open(output_file+"detailed.txt",'w')
    for res,tw,cl in zip(result,orig_test_tweets,test_tweets):
        string = str(res) + "\t"+"\t"+ str(tw.split("\t")[1])+"\t"+ str(tw.split("\t")[3])
        output.write(string)
    print "Output generated in "+output_file+"detailed.txt file\n"

    for res,tw in zip(result,orig_test_tweets):
        if res ==1:
            sarcastic_predicted_tweets = sarcastic_predicted_tweets + 1
        if res ==1 and tw.split("\t")[3][0]=="S":
            sarcastic_tweets_correctly_predicted = sarcastic_tweets_correctly_predicted + 1

    result_string="***Test Results***"+"\n"+"Total Test Tweets\t "+str(len(test_data_features))+"\n"+"Total Sarcastic Tweets present in test\t"+str(len(test_sarcasm_set))+"\n"+"Total Non Sarcastic Tweets present in test\t"+str(len(test_nonsarcasm_set))+"\n"+"Total tweets which were predicted as sarcastic\t"+str(sarcastic_predicted_tweets)+"\n"+"Total tweets which were predicted as non-sarcastic\t"+str(len(test_data_features)-sarcastic_predicted_tweets)+"\n"+"Sarcastic Tweets which were predicted correctly\t"+str(sarcastic_tweets_correctly_predicted)

    try:
        precision = float(sarcastic_tweets_correctly_predicted)/sarcastic_predicted_tweets
        recall = float(sarcastic_tweets_correctly_predicted)/total_sarcastic_tweets
        f_score = float(2*precision*recall)/(precision+recall)

        result_string = result_string + "\n\n***Final Results***" +"\n"+ "Precision\t"+str(precision) +"\n"+ "Recall\t\t"+str(recall)+"\n"+"F_score\t\t"+str(f_score)
    except:
        print "\nBad Results."
        result_string = result_string + "\nBad Results"

    print result_string
    result_stat_file = open(output_file+"stats.txt",'w')
    result_stat_file.write(result_string)

if __name__ == "__main__":
    if len(sys.argv)<5:
        print "folder, model, max_features, num, lang"
    folder=sys.argv[1]
    model=sys.argv[2]
    output=sys.argv[3]
    num = sys.argv[4]
    lang = sys.argv[5]
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    testing(folder,model,output)

