from constants import Constants
from utils import Utils
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import svds
import pandas as pd
from preprocess import Preprocess
from recommender import Recommender
from df_data import DfData
from configs import Configs
from strategies import Strategies

""" Author: Pouria Nikvand """


def main():
    # read and preprocess for the input data
    my_df_data = DfData()

    my_preprocessed = Preprocess(rating_df=my_df_data.rating_df, wishlist_df=my_df_data.wishlist_df)
    strategies = Strategies(user_id=354, my_df_data=my_df_data, my_preprocessed=my_preprocessed,
                            num_recommendations=Configs.num_recommendations)
    rus = strategies.my_recommender_2()

    # TODO handling the list
    print(rus[0])


if __name__ == '__main__':
    main()
