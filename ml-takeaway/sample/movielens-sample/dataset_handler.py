import numpy as np
import os
import pandas as pd

class DatasetHandler(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def ids2titles(self, ids):
        return ids
    
    def indices2ids(self, indices):
        return [self.movie_index_to_movie_id[index] for index in indices]
    
    def id2index(self, movieId):
        return self.movie_index_to_movie_id.index(movieId)
    
    def movie_vector2genres(self, movie_vector):
        return [self.feature_index2genre(i) for i, x in enumerate(movie_vector) if x == 1]
    
    def feature_index2genre(self, feature_index):
        return genres[feature_index]

    def load_movies(self):
        movies_frame = pd.read_csv(os.path.join(self.dataset_path, "rates.csv"),header=0,  names=["AccountId", "BookId", "Rate"], sep=",", engine="python")
        self.id_to_title = {}
        self.movie_index_to_movie_id = []
        movies_vectors = []
        for _, row in movies_frame.iterrows():
            self.id_to_title[int(row["BookId"])] = row["BookId"]
            # self.movie_index_to_movie_id.append(int(row["BookId"]))
        return np.array(movies_vectors)

    def load_users_ratings(self):
        ratings_frame = pd.read_csv(os.path.join(self.dataset_path, "rates.csv"),header=0, names=["AccountId", "BookId", "Rate"], sep=",", engine="python")
        users_ratings = {}
        for _, row in ratings_frame.iterrows():
            if int(row["AccountId"]) not in users_ratings:
                users_ratings[int(row["AccountId"])] = {}
            users_ratings[int(row["AccountId"])][int(row["BookId"])] = row["Rate"]
        return users_ratings
