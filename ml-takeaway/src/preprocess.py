import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

""" Author: Pouria Nikvand """


class Preprocess:

    def __init__(self, rating_df, wishlist_df):
        self.rating_df = rating_df
        self.wishlist_df = wishlist_df

        self.rm = rating_df.pivot(index='AccountId', columns='BookId', values='Rate')
        # self.wishlist_df['Rate'] = 1
        # self.w_rm = self.wishlist_df.pivot(index='AccountId', columns='BookId', values='Rate')

        self.account_id_indexes = pd.DataFrame(self.rm.index).reset_index(drop=True)
        # self.wishlist_account_id_indexes = pd.DataFrame(self.w_rm.index).reset_index(drop=True)

        self.rm = self.rm.reset_index(drop=True).fillna(0)
        # self.w_rm = self.w_rm.reset_index(drop=True).fillna(0)

        self.colab_onhot_filter = self.rm.where(self.rm == 0, other=1)
        self.svds_preds = self.svds_preds_building(self.rm)
        # self.wishlist_svds_preds = self.svds_preds_building(self.w_rm)
        self.nbrs_svds_preds = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nbrs_colab_onhot_filter = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nbrs_svds_preds.fit(self.svds_preds)
        self.nbrs_colab_onhot_filter.fit(self.colab_onhot_filter)

        # TODO this is time consuming
        # self.cosine_similarities_sp = self.cosine_similarity(self.svds_preds)
        # self.cosine_similarities_cof = self.cosine_similarity(self.colab_onhot_filter)

    def svds_preds_building(self, rm) -> DataFrame:
        rmm = rm.values

        mean = np.mean(rmm, axis=1)
        rmm = rmm - mean.reshape(-1, 1)

        U, sigma, Vt = svds(rmm, k=50)
        sigma = np.diag(sigma)

        x = np.matmul(np.matmul(U, sigma), Vt) + mean.reshape(-1, 1)
        preds = pd.DataFrame(x, columns=rm.columns)
        return preds

    def cosine_similarity(self, rm) -> np.ndarray:
        # TODO save and serialize the model
        A_sparse = sparse.csr_matrix(rm)
        similarities = cosine_similarity(A_sparse)
        return similarities
