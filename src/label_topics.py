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
            kl += topic[i] * math.log(
                topic[i]/(tf(word, cleaned_sentence) / len(cleaned_sentence))
                )
        except:
            continue
    return kl


def candidate_sentence_selection(topics):
    comments = clean_data.read_comments()
    sentence_candidates = {}
    for topic in topics:
        neg_kl = {}
        for comment in comments:
            for sentence in comment[0].split('.'):
                neg_kl[sentence] = -1 * kullback_liebler_divergence(topic,
                                                                    sentence)
        sentence_candidates[tuple(topic)] = sorted(neg_kl, key=neg_kl.get,
                                                   reverse=True)[:100]
    return sentence_candidates


def relevance(E, sentence_candidates, nlp):
    similarity = 0
    for sentence in sentence_candidates:
        simE = 0
        for sentenceE in E:
            simE += nlp(sentence).similarity(nlp(sentenceE))
        simS = 0
        for sentenceS in sentence_candidates:
            simS += nlp(sentence).similarity(nlp(sentenceS))
        simS *= 0.05
        similarity += min(simE, simS)
    return similarity


def coverage(E, topic, vocab):
    cover = 0
    for w, word in enumerate(vocab):
        x = 0
        for sentence in E:
            x += tf(word, clean(sentence))
        x = math.sqrt(x)
        cover += topic[w] * x
    cover *= 250
    return 0


def discrimination(E, topic, topics, vocab):
    discrimination = 0
    for topic_prime in topics:
        if tuple(topic_prime) == tuple(topic):
            continue
        for sentence in E:
            clean_sentence = clean(sentence)
            for w, word in enumerate(vocab):
                discrimination += topic[w] * tf(word, clean_sentence)
    discrimination = -300 * discrimination
    return discrimination


def summary_hueristic(E, sentence_candidates, topic, topics, vocab, nlp):
    return relevance(E, sentence_candidates, nlp) + \
        coverage(E, topic, vocab) + \
        discrimination(E, topic, topics, vocab)


def extract_summaries(L):
    vocab = np_to_sqlite.get_vocab()
    nlp = spacy.load('en')
    topics = np_to_sqlite.get_topics()
    V = candidate_sentence_selection(topics)
    topic_summaries = {}
    for topic in topics:
        U = V[tuple(topic)]
        E = []
        while len(U) > 0:
            s_hat = U[0]
            s_hat_h = 0
            Es_hat = copy.deepcopy(E)
            for sentence in U:
                Es = copy.deepcopy(E)
                Es.append(sentence)
                h = (summary_hueristic(Es, V, topic, topics, vocab, nlp) -
                     summary_hueristic(E, V, topic, topics, vocab, nlp)) / \
                    (len(sentence) ** 0.15)
                if h > s_hat_h:
                    s_hat_h = h
                    s_hat = sentence
                    Es_hat = Es
            Ewc = 0
            for sentence in E:
                Ewc += len(sentence)
            if Ewc + len(s_hat) < L and \
               summary_hueristic(Es_hat, V, topic, topics, vocab, nlp) - \
               summary_hueristic(E, V, topic, topics, vocab, nlp) > 0:
                E = Es_hat
        topic_summaries[tuple(topic)] = E
        return topic_summaries


if __name__ == '__main__':
    extract_summaries(250)
