from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from Benchmark import Benchmark
from TextAnalyzer import TextAnalyzer

__author__ = 'Raphael'


SGD_CLASSIFIER_LOSS_OPTIONS = ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',)


# REVIEW_DATA = './data/Season-1.csv'
REVIEW_DATA = './data/All-seasons.csv'

if __name__ == '__main__':
    bench = Benchmark()

    textAnalzyer = TextAnalyzer(REVIEW_DATA)
    bench.end('Initializing')

    textAnalzyer.createDataFrame(nameFilter=['Kyle', 'Stan', 'Kenny', 'Cartman', 'Butters', 'Jimmy',
                                             'Timmy'])
    bench.end('Reading CSV')

    textAnalzyer.cleanData()
    bench.end('Cleaning Data')

    textAnalzyer.splitData()
    bench.end('Generating Test and Training Data')

    textAnalzyer.determineBestParams()
    bench.end('Determining Best Results')


def determineBestParams(self):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    params = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 80),
        'clf__loss': SGD_CLASSIFIER_LOSS_OPTIONS
    }

    gridSearch = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)

    gridSearch.fit(self.dataset.X_train, self.dataset.Y_train)
    bestParams = gridSearch.best_estimator_.get_params()
    for p in sorted(params.keys()):
        print('Best param {} -> {}'.format(p, bestParams[p]))
