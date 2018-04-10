import numpy as np
import np_to_sqlite
import lda
import matplotlib.pyplot as plt


if __name__ == '__main__':
    doc_term_matrix = np_to_sqlite.get_dtm()
    vocab = np_to_sqlite.get_vocab()
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(doc_term_matrix)
    n_top_words = 8
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # plt.plot([i * 10 for i in range(150)][5:], model.loglikelihoods_[5:])
    # plt.show()
