__author__ = 'Raphael'

import re
import nltk
# nltk.download()
from nltk.corpus import stopwords

# RULE_APOSTROPHE = re.compile(r"^[^*$<,>?!']*$")

ENGLISH = "english"
POSSESSIVE_S = "\'s"
CONTRACTION_T = "'"
REF_STOP_WORDS = set(stopwords.words(ENGLISH))
DEBUG = False

class DataSanitizer(object):


    @staticmethod
    def filterWords(text):
        if  DEBUG: print 'Before Cleaning: ' + text.rstrip("\r\n")

        text = text.replace(POSSESSIVE_S, "") #remove possessive 's
        text = text.replace(CONTRACTION_T, "") #remove contraction 't
        text = re.sub("[^a-zA-Z]", " ", text) # filter out everything that is not a letter
        text = text.lower()
        words = text.split()
        words =  DataSanitizer.removeStopWords(words)
        text = " ".join(words)

        if  DEBUG: print 'After Cleaning: ' + text
        return text


    @staticmethod
    def removeStopWords(words):
        return [w for w in words if not w in REF_STOP_WORDS]


