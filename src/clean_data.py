from __future__ import absolute_import
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sqlite3
import re
import random
import numpy as np
import np_to_sqlite

# Clean comments, create document term matrix


def read_comments():
    connection = sqlite3.connect('reddit-comments.db')
    c = connection.cursor()
    c.execute('''SELECT body FROM confession''')
    comments = c.fetchall()
    connection.close()
    return comments

if __name__ == '__main__':
    # nltk.download('stopwords')
    stopwords = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    p_stemmer = PorterStemmer()
    data = read_comments()
    # random.shuffle(data)
    cleaned_data = []
    for comment in data:
        lower = comment[0].lower()
        line_token = tokenizer.tokenize(lower)
        clean_token = [re.sub(r'[^a-zA-Z]', '', word) for word in line_token]
        stop_token = [word for word in clean_token if word not in stopwords
                      if word != '']
        stem_token = [str(p_stemmer.stem(word)) for word in stop_token]
        cleaned_data.append(stem_token)
    word_list = [words for comment in cleaned_data for words in comment]
    vocab = tuple(set(word_list))
    doc_term_matrix = np.zeros((len(cleaned_data), len(vocab)),
                               dtype=np.int64)
    for i, comment in enumerate(cleaned_data):
        for word in comment:
            try:
                word_index = vocab.index(word)
                doc_term_matrix[i][word_index] += 1
            except ValueError:
                pass
    np_to_sqlite.insert_dtm(vocab, doc_term_matrix)
