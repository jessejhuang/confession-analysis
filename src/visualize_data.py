import math
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from nltk.corpus import stopwords
import clean_data
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def visualize_topics(method, n_topics):  # method True if LDA, False if LSA
    comments = [comment[0] for comment in clean_data.read_comments()]
    cvectorizer = None
    cvz = None
    lda_model = None
    X_topics = None
    n_top_words = 25  # number of keywords we show

    stop_words = stopwords.words('english')
    other_words = ['https', 'com', 'reddit', 'like',
                   'www', 'get', 'really', 'think',
                   'thing', 'someone', 'something',
                   'things', 'though', 'much', 'also',
                   'want', 'would', 'way', 'year',
                   ]
    stop_words = stop_words + other_words
    if method:
        cvectorizer = CountVectorizer(min_df=5, stop_words=stop_words)
        cvz = cvectorizer.fit_transform(comments)
        lda_model = lda.LDA(n_topics=n_topics, n_iter=1500, random_state=1)
        X_topics = lda_model.fit_transform(cvz)
    else:
        cvectorizer = TfidfVectorizer(stop_words=stop_words,
                                      use_idf=True, ngram_range=(1, 1))
        cvz = cvectorizer.fit_transform(comments)
        lda_model = TruncatedSVD(n_components=n_topics, n_iter=100)
        X_topics = lda_model.fit_transform(cvz)

    threshold = 0.4
    _idx = np.amax(X_topics, axis=1) > threshold  # idx of doc above threshold
    X_topics = X_topics[_idx]

    topic_summaries = []
    vocab = cvectorizer.get_feature_names()

    topic_word = None
    method_name = ''
    if method:
        topic_word = lda_model.topic_word_  # all topic words LDA
        method_name = 'LDA'
    else:
        topic_word = lda_model.components_  # LSA
        method_name = 'LSA'

    word_freqs = []
    for i, topic_dist in enumerate(topic_word):
        most_common_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] 
        most_common_word_freq = np.argsort(topic_dist)[:-(n_top_words + 1):-1]
        word_freq = dict(zip(most_common_words, most_common_word_freq))
        word_freqs.append(word_freq)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    
    for i in range(n_topics):
        wordcloud = WordCloud(
            background_color='white',
            mask=mask,
            width=1000,
            height=1000
        ).generate_from_frequencies(word_freqs[i])
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('topics/topic_{}.png'.format(i), bbox_inches='tight')

if __name__ == '__main__':
    visualize_topics(True, 8)