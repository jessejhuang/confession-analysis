import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import clean_data


def lsa():
    stop_words = stopwords.words('english')
    comments = [t[0] for t in clean_data.read_comments()]
    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 use_idf=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(comments)
    lsA = TruncatedSVD(n_components=5, n_iter=100)
    lsA.fit(X)
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(lsA.components_):
        termsInComp = zip(terms, comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1],
                             reverse=True)[:8]
        print("Topic %d:" % i)
        for term in sortedTerms:
            print(term[0], end=' ')
        print(" ")


if __name__ == '__main__':
    lsa()
