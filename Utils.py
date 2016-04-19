import logging

__author__ = 'Raphael'

logging.basicConfig()
logger = logging.getLogger('Utils')
logger.setLevel(logging.INFO)


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def printListItems(items):
        logger.info('\n'.join(i for i in items))