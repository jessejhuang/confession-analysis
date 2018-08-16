from __future__ import absolute_import
import os
import sqlite3
from google.cloud import bigquery


def query_reddit():
    client = bigquery.Client()
    months = ['0{}'.format(i) for i in range(1,10)] + [str(i) for i in range(10,13)]
    for month in months:
        query_job = client.query('''
            SELECT body, downs, created_utc, gilded, id, ups, rand() as rand
            FROM `fh-bigquery.reddit_comments.2017_{}`
            WHERE subreddit IN ('confession')
            ORDER BY rand
            LIMIT 1000
            '''.format(month)
            )
        results = query_job.result()  # Waits for job to complete.
        save_query(results)
        print('Saved reddit_comments.2017_{}...'.format(month))


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
        c.execute('INSERT or IGNORE INTO confession VALUES (?,?,?,?,?,?)',
                  row[0:6])
    connection.commit()
    connection.close()

def grab_comments_from_cloud():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcloud_credentials.json'
    print('Grabbing Reddit comments...')
    query_reddit()
    print('Done!')

if __name__ == '__main__':
    grab_comments_from_cloud()
