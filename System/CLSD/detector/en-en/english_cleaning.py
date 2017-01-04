#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer

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

def get_more_clean_tweet(line):
	line = re.sub(r'\[(.*?)\]','',line)
	line = line.lower()
	regex = re.compile(r'\b('+remove+r')\b',flags=re.IGNORECASE)
	line = regex.sub("",line)
	line = re.sub('(@[A-aa-z0-9]+)','',line)
	line = re.sub('(@+)','',line)
	line = re.sub('(#[A-aa-z0-9]+)','',line)
	line = re.sub('(#+)','',line)
	line = re.sub('(https://t\.co/+)','',line)
	line = re.sub('(http://t\.co/+)','',line)
	line = re.sub('(co)','',line)
	line = re.sub('(http[A-Za-z0-9://]+)','',line)
	line = re.sub('(http+)','',line)	
	return line

def get_clean_tweet(line):
	line = line.decode("utf-8",'ignore')
	try :
		data = line.split("\t")

		id = data[0]
		tweet_text = data[1]
		tweet_date = data[2]
		type = data[3]

		#remove specific words
		tweet_text = get_more_clean_tweet(tweet_text)
		# print tweet_text 
		# return
		#remove punctuation
		if(lang!='czech'):
			letters_only = re.sub("[^a-zA-Z]"," ",tweet_text)
			lower_case = letters_only.lower()        # Convert to lower case
		else:
			lower_case = letters_only=tweet_text	
		# lower_case = letters_only.lower()        # Convert to lower case
		words = lower_case.split() 
		if lang in SnowballStemmer.languages:
			stops = set(stopwords.words(lang)) 
			meaningful_words = [ w for w in words if not w in stops]

			#stem the english words
			stemmer = SnowballStemmer(lang)
			stemmed_words = [ stemmer.stem(word) for word in meaningful_words]
		else:
			stops = set(get_stop_words(lang))
			meaningful_words = [ w for w in words if not w in stops]
			stemmer = Stemmer(lang)
			# print stemmer
			stemmed_words = [ (word) for word in meaningful_words]

		#joining tweet again
		clean_tweet = " ".join(stemmed_words)
		return clean_tweet
	except (RuntimeError):
		print "Error in ", line, RuntimeError
		return ''


file1 = open("NonSarcasticTweets.txt",'r')
file2 = open("SarcasticTweets.txt",'r')
clean_tweet_file1 = open("CleanNonSarcasticTweets.txt",'w')
clean_tweet_file2 = open("CleanSarcasticTweets.txt",'w')

for line in file1:
	words = line.split("\t")
	cleansentence = get_clean_tweet(line)
	clean_tweet_file1.write(words[0]+"\t"+cleansentence+"\t"+words[2]+"\t"+words[3])
	print cleansentence


for line in file2:
	words = line.split("\t")
	cleansentence = get_clean_tweet(line)
	clean_tweet_file2.write(words[0]+"\t"+cleansentence+"\t"+words[2]+"\t"+words[3])
	print cleansentence
	