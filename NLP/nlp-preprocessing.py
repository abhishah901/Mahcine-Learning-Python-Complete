import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
import numpy as np


def create_df(corpus):
    print("Gathering data..")
    hold_files = corpus.fileids()
    rowlist = []
    for each in hold_files:
        each_row = {}
        each_row['Year'], each_row['Last_name'], _ = each.replace('-', '.').split('.')
        each_row['Text'] = pre_process(corpus.raw(each))  # Preprocessed text file
        rowlist.append(each_row)
        # print(each_row)
        # break
    print("Creating dataframe..")
    df = pd.DataFrame(rowlist)
    df['Year'] = df['Year'].astype(int)
    return df

def pre_process(text):
    # Remove punctuations
    text = "".join([str(c).lower() for c in text if c not in string.punctuation])
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    # get stemwords for each
    tokens = [PorterStemmer().stem(word) for word in tokens]
    return " ".join(tokens)




df = create_df(inaugural)




########## Year, Last name, Text