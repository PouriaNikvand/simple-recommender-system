from typing import Tuple

from utils import Utils
from preprocess import Preprocess
from constants import Constants
import pandas as pd
from pandas import DataFrame
import numpy as np

""" Author: Pouria Nikvand """


class DfData:

    def __init__(self):
        rating_path = str(Constants.constants_path.parent.parent) + Constants.rating_path
        wishlist_path = str(Constants.constants_path.parent.parent) + Constants.wishlist_path
        self.rating_df = Utils.load_data(rating_path)
        self.wishlist_df = Utils.load_data(wishlist_path)

        self.build_book_df()
        # TODO let see the distribution
        self.mean_rated_count = self.rating_df['AccountId'].value_counts().mean()
        self.max_rated_count = self.rating_df['AccountId'].value_counts().max()
        self.min_rated_count = self.rating_df['AccountId'].value_counts().min()
        self.mean_wishlist_count = self.wishlist_df['AccountId'].value_counts().mean()
        self.max_wishlist_count = self.wishlist_df['AccountId'].value_counts().max()
        self.min_wishlist_count = self.wishlist_df['AccountId'].value_counts().min()

        self.top_mean_rated_books, self.top_count_rated_books, self.top_grouped_rate = self.rate_analysis(
            self.rating_df)

        # wishlist_analysis(the_df_data.wishlist_df)


    def build_book_df(self):
        # TODO NEED TO REVIEW FOR LATER
        print(self.rating_df[self.rating_df['BookId'] == 2124])
        self.book_df_tmp = self.rating_df[['BookId', 'Rate']]
        self.book_df_tmp['Sum'] = self.book_df_tmp.groupby(['BookId'])['Rate'].transform('sum')
        self.book_df_tmp['RaCount'] = self.book_df_tmp.groupby(['BookId'])['Rate'].transform('count')

        self.book_df_tmp['NORC'] = self.book_df_tmp.groupby(['BookId', 'Rate'])['Rate'].transform('count')
        self.book_df_tmp = self.book_df_tmp.drop_duplicates()
        self.book_df_tmp['Rate1C'] = self.book_df_tmp[self.book_df_tmp['Rate'] == 1]['NORC']
        self.book_df_tmp['Rate2C'] = self.book_df_tmp[self.book_df_tmp['Rate'] == 2]['NORC']
        self.book_df_tmp['Rate3C'] = self.book_df_tmp[self.book_df_tmp['Rate'] == 3]['NORC']
        self.book_df_tmp['Rate4C'] = self.book_df_tmp[self.book_df_tmp['Rate'] == 4]['NORC']
        self.book_df_tmp['Rate5C'] = self.book_df_tmp[self.book_df_tmp['Rate'] == 5]['NORC']
        self.book_df_tmp = self.book_df_tmp.replace(np.nan, 0)
        self.book_df = self.book_df_tmp[['BookId', 'RaCount', 'Sum']].drop_duplicates()
        self.book_df_tmp = self.book_df_tmp.drop(columns=['Rate', 'NORC', 'Sum', 'RaCount'])
        self.book_df_tmp = self.book_df_tmp.groupby(['BookId'])["Rate1C", "Rate2C", "Rate3C", "Rate4C", "Rate5C"].apply(
            lambda x: x.astype(int).sum())
        self.book_df = self.book_df.merge(self.book_df_tmp, left_on='BookId', right_on='BookId')
        del self.book_df_tmp
        self.book_df = pd.DataFrame(self.book_df.values,
                                    columns=['BookId', 'RaCount', 'Sum', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C',
                                             'Rate5C'])

    @staticmethod
    def rate_analysis(rating_df) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # print(len(rating_df))
        # # 10000
        # # Checkout the data
        # print(rating_df.columns)
        # print(rating_df.head(3))

        # check for count of values in data
        print(rating_df[rating_df.duplicated()])
        # 0 data has no problem
        print(rating_df['AccountId'].nunique())
        # 6484

        # sorted by most raters
        print(rating_df['AccountId'].value_counts().head(3))

        print(rating_df['BookId'].nunique())
        # 3511

        # sorted by most rated count books
        print(rating_df['BookId'].value_counts().head(3))

        print(rating_df['Rate'].nunique())
        # 5 #[4 2 1 5 3]
        print(rating_df['Rate'].value_counts())
        print(rating_df['Rate'].describe())

        # popular top mean rated books
        grouped_rate_mean = rating_df.groupby('BookId')['Rate'].mean()
        print(grouped_rate_mean.sort_values(ascending=False).head(3))

        # popular top count rated books
        grouped_rate_count = rating_df.groupby('BookId')['Rate'].count()
        print(grouped_rate_count.sort_values(ascending=False).head(3))

        # popular top count * mean rate rated books
        grouped_rate = grouped_rate_count * grouped_rate_mean
        print(grouped_rate.sort_values(ascending=False).head(3))

        return grouped_rate_mean.index.values, grouped_rate_count.index.values, grouped_rate.index.values

    @staticmethod
    def wishlist_analysis(wishlist_df):
        # print(len(wishlist_df))
        # # 1000000

        # print(wishlist_df.columns)
        # print(wishlist_df.head(3))

        print(len(wishlist_df[wishlist_df.duplicated()]))
        # 8 data is duplicated this means 8 user are duplicated in df
        wishlist_df.drop_duplicates(inplace=True)

        print(wishlist_df['AccountId'].nunique())
        # 251753

        # most wishlist of accounts
        print(wishlist_df['AccountId'].value_counts().head(3))

        print(wishlist_df['BookId'].nunique())
        # 45890

        # most wish for books
        print(wishlist_df['BookId'].value_counts().head(3))
