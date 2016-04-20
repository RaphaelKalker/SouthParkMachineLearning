from IPython.core.display import Math

__author__ = 'Raphael'

import logging
import time

__author__ = 'Raphael'

logging.basicConfig()
logger = logging.getLogger('Benchmark')
logger.setLevel(logging.INFO)


class Benchmark:
    def __init__(self, benchmarkName=None):
        self.benchmarkName = benchmarkName
        self.__start__()

    def __start__(self):
        self.startTime = time.time()

    def end(self):
        self.endTime = time.time()
        logger.info(self.benchmarkName + " -> {}s".format("%.2f" % (self.endTime - self.startTime)))
        self.__start__()

    def end(self, benchmarkTask):
        self.endTime = time.time()
        diff = self.endTime - self.startTime

        if diff > 60:
            minutes = (diff / 60) % 60
            seconds = diff % 60
            logger.info(benchmarkTask + "{} -> {}:{} min".format(benchmarkTask, Math.round(minutes), round(seconds, 0)))
        else:
            logger.info(benchmarkTask + " -> {}s".format("%.2f" % (diff)))

        self.__start__()
