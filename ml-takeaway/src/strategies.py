from recommender import Recommender
from preprocess import Preprocess
from df_data import DfData

""" Author: Pouria Nikvand """


class Strategies(Recommender):

    def __init__(self, user_id: int, my_df_data: DfData, my_preprocessed: Preprocess, num_recommendations: int):
        super().__init__(user_id, my_df_data, my_preprocessed, num_recommendations)

    def my_recommender_1(self) -> list:
        # TODO
        results = list()

        if self.strategy_score < 5:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)
        elif self.strategy_score < 10:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)
            # results.append(self.recommend_using_svds(self.num_recommendations,
            #                                          self.my_preprocessed.wishlist_svds_preds,
            #                                          self.my_df_data.rating_df,
            #                                          self.my_preprocessed.wishlist_account_id_indexes
            #                                          ))
        elif self.strategy_score < 15:
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes
                                                     ))
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes))

            # results.append(self.recommend_using_svds(self.num_recommendations,
            #                                          self.my_preprocessed.wishlist_svds_preds,
            #                                          self.my_df_data.rating_df,
            #                                          self.my_preprocessed.wishlist_account_id_indexes
            #                                          ))
        elif self.strategy_score < 25:
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes))

        return results

    def my_recommender_2(self) -> list:
        # TODO
        results = list()

        if self.strategy_score < 5:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)
        elif self.strategy_score < 10:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 15:
            results.append(self.recommend_using_knn_users(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_svds_preds))

            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            results.append(self.recommend_using_knn_users(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_svds_preds))

        elif self.strategy_score < 25:
            results.append(self.recommend_using_knn_users(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_svds_preds))

        return results

    def my_recommender_3(self) -> list:
        # TODO
        results = list()

        if self.strategy_score < 5:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)
        elif self.strategy_score < 10:
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 15:
            results.append(self.recommend_using_knn(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_colab_onhot_filter))

            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            results.append(self.recommend_using_knn(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_colab_onhot_filter))

        elif self.strategy_score < 25:
            results.append(self.recommend_using_knn(self.num_recommendations,
                                                    self.my_preprocessed.svds_preds,
                                                    self.my_df_data.rating_df,
                                                    self.my_preprocessed.account_id_indexes,
                                                    self.my_preprocessed.nbrs_colab_onhot_filter))

        return results
