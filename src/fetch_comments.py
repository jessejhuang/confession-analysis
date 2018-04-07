from __future__ import absolute_import
import os
import sys
import sqlite3
from google.cloud import bigquery


def query_reddit():
    client = bigquery.Client()
    query_job = client.query("""
        SELECT body, downs, created_utc, gilded, id, ups, rand() as rand
        FROM `fh-bigquery.reddit_comments.2017_10`
        WHERE subreddit IN ('confession')
        ORDER BY rand
        LIMIT 200""")
    results = query_job.result()  # Waits for job to complete.
    save_query(results)


def save_query(results):
    connection = sqlite3.connect('reddit-comments.db')
    c = connection.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS confession(
            body text,
            downs integer,
            created_utc integer,
            gilded integer,
            id text PRIMARY KEY,
            ups integer
            )
        ''')
    for row in results:
        c.execute("INSERT or IGNORE INTO confession VALUES (?,?,?,?,?,?)",
                  row[0:6])
    connection.commit()
    connection.close()

if __name__ == '__main__':
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../auth-key.json"
    sys.stdout = open("log.txt", "a")
    query_reddit()
    print("2017_10")
    sys.stdout.close()
