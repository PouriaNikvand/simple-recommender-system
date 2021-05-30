from typing import Tuple

from utils import Utils
from preprocess import Preprocess
from constants import Constants
from pandas import DataFrame

""" Author: Pouria Nikvand """


class DfData:

    def __init__(self):
        rating_path = str(Constants.constants_path.parent) + Constants.rating_path
        wishlist_path = str(Constants.constants_path.parent) + Constants.wishlist_path
        self.rating_df = Utils.load_data(rating_path)
        self.wishlist_df = Utils.load_data(wishlist_path)
        self.preprocessed = Preprocess(rating_df=self.rating_df, wishlist_df=self.wishlist_df)
        self.top_mean_rated_books, self.top_count_rated_books, self.top_grouped_rate = self.rate_analysis(self.rating_df)

        # wishlist_analysis(the_df_data.wishlist_df)

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
