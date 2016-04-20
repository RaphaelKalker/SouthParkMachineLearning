from sklearn.externals import joblib
from Utils import Utils

__author__ = 'Raphael'

import logging
import pandas
import numpy as np
import time
import os
import multiprocessing
import matplotlib.pyplot as plt


from Benchmark import Benchmark
from DataSanitzer import DataSanitizer
from Dataset import Dataset

from sklearn.pipeline import Pipeline
import sknn.mlp
from sknn.backend import lasagne
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from gensim.models import Word2Vec

logging.basicConfig()
logger = logging.getLogger('TextAnalzer')
logger.setLevel(logging.INFO)

REVIEW_DATA = './data/Season-1.csv'
# REVIEW_DATA = './data/All-seasons.csv'

LINE = 'Line'
SEASON = 'Season'
EPISODE = 'Episode'
CHARACTER = 'Character'
CHARACTER_PREDICTION = 'Character Prediction'
DEBUG = True
N_FOLDS = 10
NR_FOREST_ESTIMATORS = 100
MAX_FEATURES = 10000
TEST_RATIO = 0.25
X_TRAINING = 'training_features'
Y_TRAINING = 'training_labels'
X_TESTING = 'testing_features'
Y_TESTING = 'testing_labels'
X_TRAINING_BAG_OF_WORDS = 'features_bagofwords'
X_TESTING_BAG_OF_WORDS = 'testing_features_bagofwords'
GENSIM_MODEL = 'gen_sim_model'
NAIVE_BAYES = 'Naive_Bayes'
SVM_SGD = 'SVM_SGD'
RANDOM_FOREST = 'Random_Forests'
NEURAL_NETWORK = 'Neural_Network_MLP'
NN_LEARNING_RATE = 0.002
SAVE_MODEL = False


class TextAnalyzer:
    class Data:
        def __init__(self):
            pass

    def __init__(self, csvPath):
        self.csvPath = csvPath
        self.dataFrame = None
        self.slicedDF = None
        self.dataset = Dataset()

    def createDataFrame(self, csvPath=None, nameFilter=None):
        self.dataFrame = pandas.read_csv(
            self.csvPath if csvPath is None else csvPath,
            sep=',',
            header=0,
            skipinitialspace=True,
            quotechar='"'
        )

        # names = pandas.unique(self.dataFrame[CHARACTER].values)
        # Utils.printListItems(names)

        # if True:
        #     self.dataFrame = self.dataFrame[(self.dataFrame.Character.isin(nameFilter))]

        if nameFilter:
            self.dataFrame[CHARACTER] = [self.applyNameFilter(x, nameFilter) for x in self.dataFrame.Character]

        self.dataset.X = self.dataFrame[[LINE]].values.ravel()
        self.dataset.Y = self.dataFrame[[CHARACTER]].values.ravel()

    def applyNameFilter(self, name, filter):
        return name if filter is None or name in filter else 'Other'

    # Filters out uneeded stuff and produces an array of words
    def cleanData(self):
        self.dataset.X_cleaned[:] = [DataSanitizer.filterWords(x) for x in self.dataset.X]

    def vectorizeData(self, scheme=None, resume=None):
        if scheme == 'word2vec':
            model = None
            if resume:
                model = Word2Vec.load(GENSIM_MODEL)

            if model is None:
                model = Word2Vec(
                    self.dataset.X_cleaned,
                    min_count=20,
                    size=100,
                    workers=multiprocessing.cpu_count()
                )
                model.save(GENSIM_MODEL)
        elif scheme == 'bagofwords':
            vec = CountVectorizer(
                analyzer="word",
                tokenizer=None,
                preprocessor=None,
                stop_words=None,
                max_features=MAX_FEATURES
            )

            self.dataset.X = vec.fit_transform(self.dataset.X_cleaned).toarray()

    # Normalize the frequencies with Tf-idf, this seems to shave off half the training time!
    def genTfIdf(self):
        transformer = TfidfTransformer(use_idf=True)
        transformer.fit_transform(self.dataset.X)
        self.dataset.X = transformer.transform(self.dataset.X)

    def splitData(self):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.dataset.X,  # bag of words
            self.dataset.Y,
            test_size=TEST_RATIO,
            random_state=42
        )

        self.dataset.X_train = X_train
        self.dataset.Y_train = Y_train
        self.dataset.X_test = X_test
        self.dataset.Y_test = Y_test

    def doSVMwithGridSearch(self):
        # text_clf = Pipeline([(
        #     'clf',
        #     SGDClassifier(shuffle=False, n_jobs=-1, n_iter=10, random_state=42)), ])

        parameters = {
            # 'seed': [0],
            'loss': ('log', 'hinge'),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.001, 0.0001, 0.00001, 0.000001]
        }

        classifier = GridSearchCV(SGDClassifier(), parameters, n_jobs=-1)
        return classifier
        # self.svmClf.fit(self.dataset.X_train, self.dataset.Y_train)
        # predicted = self.svmClf.predict(self.dataset.X_test)
        # self.saveResults(predicted, 'SVM new')

    def classifyData(self, algo=None, saveModel=False):
        bench = Benchmark()
        classifier = None
        prediction = None

        if algo == SVM_SGD:
            # classifier = SGDClassifier(n_jobs=-1, loss='perceptron', warm_start=True, penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
            classifier = self.doSVMwithGridSearch()
        elif algo == NEURAL_NETWORK:
            classifier = sknn.mlp.Classifier(
                layers=[ #Sigmoid, Tanh, Rectifier, Softmax, Linear
                    sknn.mlp.Layer("Tanh", units=300),
                    # sknn.mlp.Layer("Linear", units=300),
                    sknn.mlp.Layer("Softmax")],
                learning_rate=NN_LEARNING_RATE,
                n_iter=10,
                learning_momentum=.9,
                debug=False,
                regularize=None,  # L1, L2, dropout, and batch normalization.
                learning_rule='sgd'  # sgd, momentum, nesterov, adadelta, adagrad, rmsprop, adam
            )

        elif algo == RANDOM_FOREST:
            classifier = RandomForestClassifier(n_estimators=NR_FOREST_ESTIMATORS, n_jobs=-1)

        elif algo == NAIVE_BAYES:
            classifier = MultinomialNB()

        classifier.fit(self.dataset.X_train, self.dataset.Y_train)

        bench.end('Training Data using: ' + algo)

        # save that training model
        if saveModel:
            joblib.dump(classifier, './model/classifier_{}_{}'.format(algo, time.time()), compress=9)
            bench.end('Dumping Classifier Data')

        prediction = classifier.predict(self.dataset.X_test)
        bench.end('Predicting Data using: ' + algo)

        if algo == NEURAL_NETWORK:
            prediction = [x[0] for x in prediction]

        self.saveResults(prediction, algo)

    # Convenience method for printing out a panda dataframe
    def printDataFrame(self, dataframe):
        with pandas.option_context('display.max_rows', 100, 'display.max_columns', 100):
            logger.info(dataframe)

    def vectorizeDict(self):
        self.dictVec = DictVectorizer(sparse=False)
        self.dictVec.fit_transform(self.dataset.X_train)

        # self.countVec = CountVectorizer()
        # self.countVec.fit_transform(self.xTrainingData)
        # print 'Done Vectorizing'

    # def kFoldIndices(self):
    #     return KFold(n=self.dataFrame.shape[0], n_folds = N_FOLDS)

    def saveStats(self):
        frame = pandas.DataFrame(
            data={
                'Max Features': [MAX_FEATURES, ],
                'Forest Estimators': [NR_FOREST_ESTIMATORS, ],
                'Accuracy': [self.accuracy]
            }
        )

        frame = frame.transpose()

        self.printDataFrame(frame)

        frame.to_csv("./results/Trail Stats {}".format(time.time()), index=True)

        # self.printDataFrame(frame)

    def saveResults(self, prediction, classifierName):
        output = pandas.DataFrame(
            data={
                # LINE: self.dataset.X_test_original,
                CHARACTER: self.dataset.Y_test,
                CHARACTER_PREDICTION: prediction
            }
        )

        output.to_csv("{}.csv".format(classifierName), index=False)
        self.printAccuracy(self.dataset.Y_test, prediction)
        pass

    def printAccuracy(self, original, prediction):
        self.accuracy = np.mean(original == prediction)
        logger.info('Accuracy: {} %'.format(round(self.accuracy * 100, 3)))

    def optimizeParams(self):
        self.params = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-1, 1e-2, 1e-3),
        }




bench = Benchmark()

anal = TextAnalyzer(REVIEW_DATA)
bench.end('Initializing')

anal.createDataFrame(nameFilter=['Kyle', 'Stan', 'Kenny', 'Cartman', 'Butters', 'Jimmy',
                                 'Timmy'])
bench.end('Reading CSV')

anal.cleanData()  # Prepare data in a format that is good for scikitlearn
bench.end('Cleaning Data')

# anal.vectorizeData(scheme='word2vec')
# bench.end('Creating Word 2 Vec')

anal.vectorizeData(scheme='bagofwords')
bench.end('Generating Bag of Words Representation')

anal.genTfIdf()  #
bench.end('Generating TF-IDF Representation')

anal.splitData()
bench.end('Generating Test and Training Data')

# plt.scatter(anal.dataset.X_train, anal.dataset.Y_train, color='black')
# plt.show

# anal.optimizeParams()

anal.classifyData(NAIVE_BAYES, saveModel=SAVE_MODEL)

# anal.doSVM()

# anal.saveStats()
