import re

from nltk.stem.snowball import SnowballStemmer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer

lang = 'english'
#dictionnary to sentiment analysis
emo_repl = {
    #good emotions
    "&lt;3" : " good ",
    ":d" : " good ",
    ":dd" : " good ",
    ":p" : " good ",
    "8)" : " good ",
    ":-)" : " good ",
    ":)" : " good ",
    ";)" : " good ",
    "(-:" : " good ",
    "(:" : " good ",
    
    "yay!" : " good ",
    "yay" : " good ",
    "yaay" : " good ",
    "yaaay" : " good ",
    "yaaaay" : " good ",
    "yaaaaay" : " good ",    
    #bad emotions
    ":/" : " bad ",
    ":&gt;" : " sad ",
    ":')" : " sad ",
    ":-(" : " bad ",
    ":(" : " bad ",
    ":s" : " bad ",
    ":-s" : " bad "
}

#dictionnary for general (i.e. topic modeler)
emo_repl2 = {
    #good emotions
    "&lt;3" : " heart ",
    ":d" : " smile ",
    ":p" : " smile ",
    ":dd" : " smile ",
    "8)" : " smile ",
    ":-)" : " smile ",
    ":)" : " smile ",
    ";)" : " smile ",
    "(-:" : " smile ",
    "(:" : " smile ",
       
    #bad emotions
    ":/" : " worry ",
    ":&gt;" : " angry ",
    ":')" : " sad ",
    ":-(" : " sad ",
    ":(" : " sad ",
    ":s" : " sad ",
    ":-s" : " sad "
}


#general
re_repl = {
    r"\br\b" : " are ",
    r"\bu\b" : " you ",
    r"\bhaha\b" : " ha ",
    r"\bhahaha\b" : " ha ",
    r"\bdon't\b" : " do not ",
    r"\bdoesn't\b" : " does not ",
    r"\bdidn't\b" : " did not ",
    r"\bhasn't\b" : " has not ",
    r"\bhaven't\b" : " have not ",
    r"\bhadn't\b" : " had not ",
    r"\bwon't\b" : " will not ",
    r"\bwouldn't\b" : " would not ",
    r"\bcan't\b" : " can not ",
    r"\bcannot\b" : " can not ",
    r"\!" : " exclamation ",
    r"\?" : " question ",
    r"http[^\t\n ]*":"",
    r"\[[^\t\n\]]*\]": "",
    r"\@[^\t\n ]*":"",
    r"\#[^\t\n ]*":"",
    r"sarcasm" : "",
    r"sarcastic" : "",
    r"\tRT " : "\t",
    r"\trt " : "\t",
}

emo_repl_order = [k for (k_len,k) in reversed(sorted([(len(k),k) for k in emo_repl.keys()]))]
emo_repl_order2 = [k for (k_len,k) in reversed(sorted([(len(k),k) for k in emo_repl2.keys()]))]

def stem_sentence(tweet_text,lang):
    try:
        lower_case = tweet_text.decode('utf-8','ignore').lower() 

        # Convert to lower case
        words = lower_case.split(" ") 
        if lang in SnowballStemmer.languages:
            #stem the english words
            stemmer = SnowballStemmer(lang)
            stemmed_words = [ stemmer.stem(word) for word in words]
        else:
            stemmer = Stemmer(lang)
            stemmed_words = [ (word) for word in words]
        return " ".join(stemmed_words)    
    except:
        print "error in tweet", tweet_text
        exit(0)

def replace_emo(sentence):
    sentence2 = sentence
    for k in emo_repl_order:
        sentence2 = sentence2.replace(k,emo_repl[k])
    for r, repl in re_repl.iteritems():
        sentence2 = re.sub(r,repl,sentence2)
    return sentence2

def replace_reg(sentence):
    sentence2 = sentence
    for r, repl in re_repl.iteritems(): 
        sentence2 = re.sub(r,repl,sentence2)
    for k in emo_repl_order2:
        sentence2 = sentence2.replace(k,emo_repl2[k])
    return sentence2



def get_clean_tweet(line):
    s =  stem_sentence(replace_emo(replace_reg(line.split("\t")[1])),lang)
    return s
    return ((line.split("\t")[1]))

par = True 