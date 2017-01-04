#!/bin/sh
head -n $1 $2/SarcasticTweets.txt > $2/TestSarcasticTweets.txt
head -n $1 $2/NonSarcasticTweets.txt > $2/TestNonSarcasticTweets.txt
