'''
Implemtation of Text summary automatic labeling described at
https://pdfs.semanticscholar.org/4cc4/898e86c4e9d2a87bbb26f18ec6ebb79e9ded.pdf
'''
import re
import math
import copy
import np_to_sqlite
import clean_data
import nltk
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def tf(word, sentence):
    wc = sentence.count(word)
    if(wc > 0):
        return wc
    return 10e-6


def clean(sentence):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    p_stemmer = PorterStemmer()
    sentence = tokenizer.tokenize(sentence.lower())
    clean_token = [re.sub(r'[^a-zA-Z]', '', word) for word in sentence]
    stop_token = [word for word in clean_token if word not in stop_words
                  if word != '']
    stem_token = [str(p_stemmer.stem(word)) for word in stop_token]
    return stem_token


def kullback_liebler_divergence(topic, sentence):
    cleaned_sentence = clean(sentence)
    vocab = np_to_sqlite.get_vocab()
    kl = 0
    for word in cleaned_sentence:
        try:
            w = vocab.index(word)
            kl += topic[1][w] * math.log(
                topic[1][w]/(tf(word,
                                cleaned_sentence) / len(cleaned_sentence))
                )
        except:
            continue
    return kl


def candidate_sentence_selection():
    """ Determine the 100 candidate sentences for each topic
    """
    comments = clean_data.read_comments()
    sentence_candidates = {}
    topics = np_to_sqlite.get_topics()
    # topic is a (topic_id, topic) tuple
    for topic in topics:
        neg_kl = {}
        for comment in comments:
            for sentence in comment[0].split('.'):
                neg_kl[sentence] = -1 * kullback_liebler_divergence(topic,
                                                                    sentence)
        # 100 candidate sentences per topic
        sentence_candidates[topic[0]] = sorted(neg_kl, key=neg_kl.get,
                                               reverse=True)[:100]
    np_to_sqlite.store_sentence_candidates(sentence_candidates)


def relevance(E, topic, sentence_candidates, nlp):
    print("In relevance")
    similarity = 0
    topic_sentences = sentence_candidates
    for sentence_tuple in topic_sentences:
        sentence = sentence_tuple[0]
        simE = 0
        for sentenceE in E:
            simE += nlp(sentence).similarity(nlp(sentenceE))
        simV = 0
        for sentenceV_tuple in topic_sentences:
            sentenceV = sentenceV_tuple[0]
            simV += nlp(sentence).similarity(nlp(sentenceV))
        simV *= 0.05
        similarity += min(simE, simV)
    return similarity


def coverage(E, topic_id, topics, vocab):
    print("In coverage")
    cover = 0
    for w, word in enumerate(vocab):
        x = 0
        for sentence in E:
            x += tf(word, clean(sentence))
        x = math.sqrt(x)
        cover += topics[topic_id - 1][1][w] * x
    cover *= 250
    return 0


def discrimination(E, topic_id, topics, vocab):
    discrimination = 0
    # for (topic_id_prime, topic_prime) in topics:
    #     if topic_id == topic_id_prime:
    #         continue
    #     for sentence in E:
    #         clean_sentence = clean(sentence)
    #         for w, word in enumerate(vocab):
    #             discrimination += topic_prime[w] * tf(word, clean_sentence)
    # discrimination = -300 * discrimination
    return discrimination


def summary_hueristic(E, sentence_candidates, topic_id, topics, vocab, nlp):
    return relevance(E, topic_id, sentence_candidates, nlp) + \
        coverage(E, topic_id, topics, vocab) + \
        discrimination(E, topic_id, topics, vocab)


def extract_summaries(L):
    vocab = np_to_sqlite.get_vocab()
    nlp = spacy.load('en')
    topics = np_to_sqlite.get_topics()
    topic_summaries = {}
    for (topic_id, topic) in topics:
        V = np_to_sqlite.get_sentence_candidates(topic_id)
        U = copy.deepcopy(V)
        E = []
        print("Topic: ", topic_id)
        while len(U) > 0:
            # s_hat, optimal sentence, initially set to first candidate
            s_hat = U[0][0]
            s_hat_h = 0
            Es_hat = copy.deepcopy(E)
            for sentence_tuple in U:
                sentence = sentence_tuple[0]
                print("Sentence: ", sentence)
                if len(sentence_tuple) < 1:
                    continue
                Es = copy.deepcopy(E)
                Es.append(sentence)
                h = (summary_hueristic(Es, V, topic_id, topics, vocab, nlp) -
                     summary_hueristic(E, V, topic_id, topics, vocab, nlp)) / \
                    (len(sentence.split(' ')) ** 0.15)
                if h > s_hat_h:
                    s_hat_h = h
                    s_hat = sentence
                    Es_hat = Es
            Ewc = 0
            for sentence in E:
                Ewc += len(sentence.split(' '))
            if Ewc + len(s_hat) < L and \
               summary_hueristic(Es_hat, V, topic_id, topics, vocab, nlp) - \
               summary_hueristic(E, V, topic_id, topics, vocab, nlp) > 0:
                E = Es_hat
            U.remove(s_hat)
            print("Length of U: ", len(U))
        topic_summaries[topic_id] = E
    return topic_summaries


if __name__ == '__main__':
    extract_summaries(250)
