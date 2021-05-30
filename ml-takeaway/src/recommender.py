from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from termcolor import colored

""" Author: Pouria Nikvand """


class Recommender:

    def __init__(self, user_id, rating_df, wishlist_df):
        self.user_id = user_id
        self.user_wishlist_books = wishlist_df[wishlist_df['AccountId'] == self.user_id]
        self.user_rated_books = rating_df[rating_df['AccountId'] == self.user_id]

    def cosine_similarity(self, rating_matrix):
        pass

    def recommend_using_knn(self):
        pass

    def recommend_using_svds(self, num_recommendations, preds, rating_df, account_id_indexes) -> Tuple[DataFrame, bool]:
        try:
            user_index = account_id_indexes[account_id_indexes['AccountId'] == self.user_id].index.values[0]
        except IndexError:
            print(colored('This Account ID is not in the source of our data', 'blue', attrs=['bold']))
            return pd.DataFrame(), False
        sorted_user_predictions = preds.iloc[user_index].sort_values(ascending=False)
        books = pd.DataFrame(rating_df.BookId.unique(), columns=['BookId'])
        recommendations = (
            books[~books.isin(self.user_rated_books)].merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                                            how='left',
                                                            left_on='BookId',
                                                            right_on='BookId').rename(
                columns={user_index: 'Predictions'}))
        result = recommendations.sort_values('Predictions', ascending=False).iloc[:num_recommendations, :-1]
        return result, True
