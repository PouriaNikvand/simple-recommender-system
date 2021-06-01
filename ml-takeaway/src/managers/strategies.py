from managers.recommender import Recommender
from managers.preprocess import Preprocess
from managers.df_data import DfData

""" Author: Pouria Nikvand """


class Strategies(Recommender):

    def __init__(self, user_id: int, my_df_data: DfData, my_preprocessed: Preprocess, num_recommendations: int):
        super().__init__(user_id, my_df_data, my_preprocessed, num_recommendations)

    def my_recommender_1(self) -> list:
        # using svds preds for prediction with strategy for new users

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

        elif self.strategy_score < 25:
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes))

        return results

    def my_recommender_2(self) -> list:
        # using knn users on svds preds for prediction with svds
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
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_svds_preds)
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))

            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_svds_preds)

            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))

        elif self.strategy_score < 25:
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_svds_preds)
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))
        return results

    def my_recommender_3(self) -> list:
        # using knn users on onehot rm matrix preds for prediction with svds
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
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_colab_onhot_filter)
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_colab_onhot_filter)

            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))
        elif self.strategy_score < 25:
            similar_users = self.recommend_using_knn_users(self.num_recommendations,
                                                           self.my_preprocessed.svds_preds,
                                                           self.my_df_data.rating_df,
                                                           self.my_preprocessed.account_id_indexes,
                                                           self.my_preprocessed.nbrs_colab_onhot_filter)
            results.append(self.recommend_using_svds(self.num_recommendations,
                                                     self.my_preprocessed.svds_preds,
                                                     self.my_df_data.rating_df,
                                                     self.my_preprocessed.account_id_indexes,
                                                     similar_users))
        return results

    def my_recommender_4(self, my_deep_model) -> list:
        # using deep model trained on user rated data for prediction
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
            results.append(my_deep_model.test(self.user_id, self.num_recommendations))
            results.append(self.my_df_data.top_count_rated_books)
            results.append(self.my_df_data.top_mean_rated_books)
            results.append(self.my_df_data.top_grouped_rate)

        elif self.strategy_score < 20:
            results.append(my_deep_model.test(self.user_id, self.num_recommendations))
        elif self.strategy_score < 25:
            results.append(my_deep_model.test(self.user_id, self.num_recommendations))

        return results
