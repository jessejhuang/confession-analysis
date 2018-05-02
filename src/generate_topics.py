import numpy as np
import np_to_sqlite
import lda
# import matplotlib.pyplot as plt


def generate_topics():
    doc_term_matrix = np_to_sqlite.get_dtm()
    vocab = np_to_sqlite.get_vocab()
    model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
    model.fit(doc_term_matrix)
    topic_word = model.topic_word_
    # np_to_sqlite.store_topics(topic_word)
    return model


def print_topics():
    n_top_words = 8
    model = generate_topics()
    topic_word = model.topic_word_
    vocab = np_to_sqlite.get_vocab()
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

if __name__ == '__main__':
    print_topics()
    # for i, topic_dist in enumerate(topic_word):
    #     topic_words =
    #                   np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # plt.plot([i * 10 for i in range(150)][5:], model.loglikelihoods_[5:])
    # plt.show()
