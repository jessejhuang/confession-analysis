'''
Store np array into sqlite table
See https://stackoverflow.com/a/18622264
'''
import sqlite3
import numpy as np
import io


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def insert_dtm(vocab, x):
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect("reddit-comments.db",
                          detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS dtm")
    cur.execute("CREATE TABLE dtm (arr array)")
    cur.execute("INSERT INTO dtm (arr) VALUES (?)", (x,))
    con.commit()
    con.close()
    # Insert vocab
    con = sqlite3.connect("reddit-comments.db")
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS vocab")
    cur.execute('''
        CREATE TABLE vocab(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT
            )
        ''')
    for word in vocab:
        cur.execute("INSERT INTO vocab (word) values(?)", [word])
    con.commit()
    con.close()


def get_dtm():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect("reddit-comments.db",
                          detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("SELECT arr FROM dtm")
    dtm = cur.fetchone()[0]
    con.close()
    return dtm


def get_vocab():
    con = sqlite3.connect("reddit-comments.db")
    cur = con.cursor()
    cur.execute("SELECT word FROM vocab ORDER BY id")
    data = cur.fetchall()
    con.close()
    vocab = []
    for word in data:
        vocab.append(word[0])
    return tuple(vocab)


def store_topics(topics):
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect("reddit-comments.db",
                          detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS topics")
    cur.execute("CREATE TABLE topics (topic array)")
    cur.execute("INSERT INTO topics (topic) values(?)", (topics,))
    con.commit()
    con.close()


def get_topics():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect("reddit-comments.db",
                          detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("SELECT topic FROM topics")
    topics = cur.fetchone()[0]
    con.close()
    return topics
