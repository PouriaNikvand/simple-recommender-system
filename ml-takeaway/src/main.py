from termcolor import colored

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
from deep_model import DeepModel

""" Author: Pouria Nikvand """


def main():
    # read and preprocess for the input data
    my_df_data = DfData()

    # TODO a deep model added for rate prediciton for a user
    # this could act like a validator or a new model for the system
    # with this model we can predict all the rates for each user based on books and user behavior
    my_deep_model = DeepModel(my_df_data)

    my_preprocessed = Preprocess(rating_df=my_df_data.rating_df, wishlist_df=my_df_data.wishlist_df)
    strategies = Strategies(user_id=354, my_df_data=my_df_data, my_preprocessed=my_preprocessed,
                            num_recommendations=Configs.num_recommendations)

    rus1 = strategies.my_recommender_1()
    rus2 = strategies.my_recommender_2()
    rus3 = strategies.my_recommender_3()
    rus4 = strategies.my_recommender_4(my_deep_model)

    print(colored('the results of each strategies is : ', 'red'))
    print(colored('the results of each strategies is : ', 'red'))
    print(colored('first strategy : ', 'blue'))
    print(colored(rus1, 'blue'))
    print(colored('second strategy : ', 'yellow'))
    print(colored(rus2, 'yellow'))
    print(colored('third strategy : ', 'green'))
    print(colored(rus3, 'green'))
    print(colored('fourth strategy : ', 'magenta'))
    print(colored(rus4, 'magenta'))

if __name__ == '__main__':
    main()
