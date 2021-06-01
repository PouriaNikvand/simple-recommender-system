from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from termcolor import colored
from df_data import DfData
from preprocess import Preprocess

""" Author: Pouria Nikvand """


class Recommender():

    def __init__(self, user_id: int, my_df_data: DfData, my_preprocessed: Preprocess, num_recommendations: int):
        self.user_id = user_id
        self.my_df_data = my_df_data
        self.num_recommendations = num_recommendations
        self.my_df_data = my_df_data
        self.my_preprocessed = my_preprocessed
        self.user_wishlist_books = my_df_data.wishlist_df[my_df_data.wishlist_df['AccountId'] == self.user_id]
        self.user_rated_books = my_df_data.rating_df[my_df_data.rating_df['AccountId'] == self.user_id]

        # TODO we can let the user interpolate with the sterategy
        self.strategy_score = self.strategy_picker()

    def strategy_picker(self) -> int:
        # TODO
        # if the user is not a common user and his/her rate is not normal
        # if unusual likes / unusual dislikes / unique users and items

        # define strategy based on user personalized data
        # if the user is very unkknown to us and we don't know about him/her
        # we can use mean or something instead of 0 like a num comes out of the normal distribution a user books count
        if len(self.user_rated_books) == 0:
            if len(self.user_wishlist_books) == 0:
                return 1
            return 5

        # if the user is a good user
        elif len(self.user_wishlist_books) > 0:
            if len(self.user_rated_books) >= self.my_df_data.mean_rated_count:
                return 20
            return 15
        return 10

    def recommend_using_knn_users(self, num_recommendations, user_vec_profiles, rating_df, account_id_indexes,
                                  nbrs) -> list:

        user_index = account_id_indexes[account_id_indexes['AccountId'] == self.user_id].index.values[0]
        user_vec_profiles = user_vec_profiles.iloc[user_index]
        my_vec = user_vec_profiles.values.reshape(1, -1)
        nearest_users_index = [i for i in
                               nbrs.kneighbors(my_vec, n_neighbors=num_recommendations, return_distance=True)]
        nearest_users = account_id_indexes.iloc[nearest_users_index[1][0]]['AccountId']
        return nearest_users.iloc[1:].values

    def recommend_using_svds(self, num_recommendations, preds, rating_df, account_id_indexes, user_id=None) -> Tuple[
        DataFrame, bool]:
        result = []
        if user_id is None:
            user_id = self.user_id
            user_wishlist_books = self.user_wishlist_books
            user_rated_books = self.user_rated_books
            # TODO we had a problem in the original code that has been fixed here / ask later for checking this
            user_index = account_id_indexes[account_id_indexes['AccountId'] == user_id].index.values[0]
            sorted_user_predictions = preds.iloc[user_index].sort_values(ascending=False)
            books = pd.DataFrame(rating_df.BookId.unique(), columns=['BookId'])
            recommendations = (
                books[~books.isin(user_rated_books)].merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                                           how='left',
                                                           left_on='BookId',
                                                           right_on='BookId').rename(
                    columns={user_index: 'Predictions'}))
            result = recommendations.sort_values('Predictions', ascending=False).iloc[:num_recommendations, :-1]
        else:
            if not isinstance(user_id, int):
                users_recoms = pd.DataFrame([])
                for each_user in user_id:
                    user_rated_books = self.my_df_data.rating_df[self.my_df_data.rating_df['AccountId'] == each_user]
                    user_index = account_id_indexes[account_id_indexes['AccountId'] == each_user]
                    user_index = user_index.index.values[0]
                    sorted_user_predictions = preds.iloc[user_index].sort_values(ascending=False)
                    books = pd.DataFrame(rating_df.BookId.unique(), columns=['BookId'])
                    recommendations = (
                        books[~books.isin(user_rated_books)].merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                                                   how='left',
                                                                   left_on='BookId',
                                                                   right_on='BookId').rename(
                            columns={user_index: 'Predictions'}))
                    users_recoms = pd.concat([recommendations, users_recoms], axis=0)
                result = users_recoms.sort_values('Predictions', ascending=False).iloc[:, :-1]
                result = result.drop_duplicates().iloc[:num_recommendations]
        return result
