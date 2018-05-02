import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import clean_data
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool


def visualize_topics(s, n_topics):  # s True if LDA, False if LSA
    comments = [comment[0] for comment in clean_data.read_comments()]
    method = False
    if s == 'True':
        method = True
    cvectorizer = None
    cvz = None
    lda_model = None
    X_topics = None
    n_top_words = 8  # number of keywords we show

    if method:
        cvectorizer = CountVectorizer(min_df=5, stop_words='english')
        cvz = cvectorizer.fit_transform(comments)
        lda_model = lda.LDA(n_topics=n_topics, n_iter=1500, random_state=1)
        X_topics = lda_model.fit_transform(cvz)
    else:
        cvectorizer = TfidfVectorizer(stop_words='english',
                                      use_idf=True, ngram_range=(1, 1))
        cvz = cvectorizer.fit_transform(comments)
        lda_model = TruncatedSVD(n_components=n_topics, n_iter=100)
        X_topics = lda_model.fit_transform(cvz)

    threshold = 0.5
    _idx = np.amax(X_topics, axis=1) > threshold  # idx of doc above threshold
    X_topics = X_topics[_idx]

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                      init='pca')
    tsne_lda = tsne_model.fit_transform(X_topics)

    # 20 colors
    colormap = np.array([
        "#1f77b4", "#aec7e8",  "#ffbb78", "#2ca02c", "#c5b0d5",
        "#98df8a", "#ff7f0e",  "#d62728", "#ff9896", "#9467bd",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])
    _lda_keys = []
    for i in range(X_topics.shape[0]):
        _lda_keys += X_topics[i].argmax(),
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

    for i, topic_dist in enumerate(topic_word):
        topic_words = \
            np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_summaries.append(' '.join(topic_words))  # append!
    title = 'Confident Comment Visualization with {} topics using {}'.format(
        n_topics, method_name)
    num_example = len(X_topics)

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                         title=title,
                         tools='''pan,wheel_zoom,box_zoom,reset,hover,
                         previewsave''',
                         x_axis_type=None, y_axis_type=None, min_border=1)
    source = bp.ColumnDataSource({
        "x": np.array(tsne_lda[:, 0]),
        "y": np.array(tsne_lda[:, 1]),
        "color": colormap[_lda_keys][:num_example],
        "content": comments[:num_example],
        "topic_key": np.array(_lda_keys[:num_example]),
    })
    plot_lda.scatter(
                     x='x',
                     y='y',
                     source=source,
                     color='color'
                     )
    topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
    for topic_num in _lda_keys:
        if not np.isnan(topic_coord).any():
            break
        topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

    for i, topic in enumerate(topic_coord):
        for j, t in enumerate(topic):
            if np.isnan(t):
                topic_coord[i][j] = np.float64(0)
    # plot crucial words
    for i in range(X_topics.shape[1]):
        plot_lda.text(
            topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

    # hover tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"content": "@content - topic: @topic_key"}

    # save the plot
    save(plot_lda, '{}.html'.format(title))
