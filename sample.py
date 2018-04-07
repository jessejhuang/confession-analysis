from __future__ import absolute_import
import os
import sys
import sqlite3
from google.cloud import bigquery


def query_stackoverflow():
    client = bigquery.Client()
    query_job = client.query("""
        SELECT
          CONCAT(
            'https://stackoverflow.com/questions/',
            CAST(id as STRING)) as url,
          view_count
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        WHERE tags like '%google-bigquery%'
        ORDER BY view_count DESC
        LIMIT 10""")

    results = query_job.result()  # Waits for job to complete.
    save_query(results)


def save_query(results):
    connection = sqlite3.connect('stack-overflow.db')
    c = connection.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
        (url text, view_count integer)''')
    for row in results:
        c.execute(
            "INSERT INTO posts VALUES (?,?)", [row.url, row.view_count])
    connection.commit()
    connection.close()

if __name__ == '__main__':
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth-key.json"
    sys.stdout = open("query.txt", "w")
    query_stackoverflow()
    sys.stdout.close()
