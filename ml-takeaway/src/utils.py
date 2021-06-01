import pandas as pd
import pickle
import numpy as np


""" Author: Pouria Nikvand """


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def load_model():
        pass
