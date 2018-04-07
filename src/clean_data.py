from __future__ import absolute_import
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sqlite3
import re
import random


def read_comments():
    connection = sqlite3.connect('reddit-comments.db')
    c = connection.cursor()
    c.execute('''SELECT body FROM confession LIMIT 20''')
    comments = c.fetchall()
    connection.close()
    return comments

if __name__ == '__main__':
    # nltk.download('stopwords')
    stopwords = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    p_stemmer = PorterStemmer()
    data = read_comments()
    random.shuffle(data)
    for comment in data:
        lower = comment[0].lower()
        line_token = tokenizer.tokenize(lower)
        clean_token = [re.sub(r'[^a-zA-Z]', '', word) for word in line_token]
        stop_token = [word for word in clean_token if word not in stopwords
                      if word != '']
        stem_token = [str(p_stemmer.stem(word)) for word in stop_token]
        print(stem_token)
