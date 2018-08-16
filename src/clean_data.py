import sqlite3
# Get comments from sqlite

def read_comments():
    connection = sqlite3.connect('reddit-comments.db')
    c = connection.cursor()
    c.execute('''SELECT body FROM confession''')
    comments = c.fetchall()
    connection.close()
    return comments
