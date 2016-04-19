__author__ = 'Raphael'


class Dataset:


        def __init__(self):
            self.X = []
            self.Y = []

            self.X_cleaned = []


            self.Y_test_original = []
            self.X_trainCleaned = []
            self.X_testCleaned = []


            self.X_train = []
            self.Y_train = []

            self.X_test = []

            self.Y_test = []
            self.X_test_bagofwords = []

            self.X_train_original = []
            self.X_test_original = []

            self.xTrainingCleaned = []
            self.bagOfWordsFeatures = []

        # @property
        # def X_train(self):
        #     return self.X_train
        #
        # @X_train.setter
        # def X_train(self, value):
        #     self.X_train = value

        # @property
        # def Y_train(self):
        #     return self.Y_train
        #
        # @Y_train.setter
        # def Y_train(self, value):
        #     self.Y_train = value
        #
        # # @property
        # # def xTrainingCleaned(self):
        # #     return self.xTrainingCleaned
        # #
        # # @xTrainingCleaned.setter
        # # def xTrainingCleaned(self, value):
        # #     self.xTrainingCleaned = value
        # #     pass
        #
        # @property
        # def bagOfWordsFeatures(self):
        #     return self.bagOfWordsFeatures
        #
        # @bagOfWordsFeatures.setter
        # def bagOfWordsFeatures(self, value):
        #     self.bagOfWordsFeatures = value
        #     pass
        #
        # @property
        # def X_test(self):
        #     return self.X_test
        #
        # # @X_test.setter
        # # def X_test(self, value):
        # #     self.X_test = value
        # #
        # # @property
        # # def Y_test(self):
        # #     return self.Y_test
        #
        # @Y_test.setter
        # def Y_test(self, value):
        #     self.Y_test = value
