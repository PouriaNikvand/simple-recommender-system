import pandas as pd

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
