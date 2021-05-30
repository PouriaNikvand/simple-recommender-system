from constants import Constants
from utils import Utils
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import svds
import pandas as pd
from preprocess import Preprocess
from recommender import Recommender
from df_data import DfData

""" Author: Pouria Nikvand """


def main():
    # read and preprocess for the input data
    sample_data = DfData()
    num_recommendations = 10
    recommender = Recommender(user_id=31154, rating_df=sample_data.rating_df, wishlist_df=sample_data.wishlist_df)
    rus, res_flag = recommender.recommend_using_svds(num_recommendations=num_recommendations,
                                                     preds=sample_data.preprocessed.svds_preds,
                                                     rating_df=sample_data.rating_df,
                                                     account_id_indexes=sample_data.preprocessed.account_id_indexes)
    if res_flag:
        print(rus)
    else:
        return print(sample_data.top_grouped_rate[:num_recommendations])


if __name__ == '__main__':
    main()
