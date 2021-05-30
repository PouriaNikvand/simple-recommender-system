import os
from pathlib import PurePosixPath

""" Author: Pouria Nikvand """


class Constants:
    constants_path = PurePosixPath(os.path.dirname(os.path.abspath(__file__)))
    rating_path = '/dataset/ml-takeaway/ratings.csv'
    wishlist_path = '/dataset/ml-takeaway/wishlist.csv'
