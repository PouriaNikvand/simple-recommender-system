import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import PurePosixPath
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Flatten, Activation, Input, Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model
from df_data import DfData

""" Author: Pouria Nikvand """


class DeepModel:

    def __init__(self, my_df_data: DfData):
        books, users, user_rating = self.preprocess(my_df_data.book_df, my_df_data.rating_df)
        self.BRS(books, users, user_rating)

    def preprocess(self, book_rating, user_rating):

        book_rating['Rate1C'] = book_rating['Rate1C'].astype(int)
        book_rating['Rate2C'] = book_rating['Rate2C'].astype(int)
        book_rating['Rate3C'] = book_rating['Rate3C'].astype(int)
        book_rating['Rate4C'] = book_rating['Rate4C'].astype(int)
        book_rating['Rate5C'] = book_rating['Rate5C'].astype(int)

        book_rating['Pct_1Star'] = book_rating['Rate1C'] / book_rating['RaCount']
        book_rating['Pct_2Star'] = book_rating['Rate2C'] / book_rating['RaCount']
        book_rating['Pct_3Star'] = book_rating['Rate3C'] / book_rating['RaCount']
        book_rating['Pct_4Star'] = book_rating['Rate4C'] / book_rating['RaCount']
        book_rating['Pct_5Star'] = book_rating['Rate5C'] / book_rating['RaCount']
        book_rating.head()

        book_rating_df = book_rating[['BookId', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']]
        scaling_cols = ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum', 'RaCount']

        book_rating_scaled = self.mix_max_scaler(book_rating, scaling_cols)
        book_rating_df = book_rating_scaled[
            ['BookId', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum', 'RaCount']]

        ##Let's create Book_id that we can use
        book_id_0 = book_rating_df[['BookId']]
        book_id_1 = user_rating[['BookId']]
        book_id = pd.concat([book_id_0, book_id_1], axis=0, ignore_index=True)
        book_id.rename(columns={book_id.columns[0]: "BookId"}, inplace=True)
        book_id.drop_duplicates(inplace=True)
        # book_id['Book_Id'] = book_id.index.values
        book_id.head()

        user_rating.head(3)
        user_rating = pd.merge(user_rating, book_id, on='BookId', how='left')
        book_rating_df = pd.merge(book_rating_df, book_id, on='BookId', how='left')
        user_rating.head()
        user_rating['Rate'].unique()

        book_rating_numeric = book_rating_df[
            ['BookId', 'RaCount', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum']]
        user_rating = pd.merge(user_rating, book_rating_numeric, on='BookId', how='left')
        user_rating.head()

        user_rating.fillna(0, inplace=True)

        users = user_rating.AccountId.unique()
        books = user_rating.BookId.unique()

        return books, users, user_rating

    def BRS(self, books, users, user_rating):
        userid2idx = {o: i for i, o in enumerate(users)}
        bookid2idx = {o: i for i, o in enumerate(books)}
        user_rating['AccountId'] = user_rating['AccountId'].apply(lambda x: userid2idx[x])
        user_rating['BookId'] = user_rating['BookId'].apply(lambda x: bookid2idx[x])
        self.y = user_rating['RaCount']
        self.X = user_rating.drop(['RaCount'], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # this part as for production not for experiments
        # at this part we have done the experiments enough
        self.X_train, self.y_train = self.X, self.y

        print(self.X_train.shape, self.X_test.shape)

        self.exogenous_train = np.array(self.X_train[
                                       ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum']])
        self.exogenous_valid = np.array(self.X_test[
                                       ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum']])

        n_books = len(user_rating['BookId'].unique())
        n_users = len(user_rating['AccountId'].unique())

        self.nn_model = self.build_model(0.4, 65, n_books, n_users)
        self.nn_model.summary()

        self.nn_model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        # we always want this batch size better than the others :-D
        batch_size = 32
        epochs = 10
        History = self.nn_model.fit([self.X_train.AccountId, self.X_train.BookId, self.exogenous_train], self.y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(
                                    [self.X_test.AccountId, self.X_test.BookId, self.exogenous_valid], self.y_test),
                                    verbose=1)

        plt.plot(History.history['loss'], 'g')
        plt.plot(History.history['val_loss'], 'b')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.show()

        # self.result_analysis()

    def result_analysis(self):
        preds = self.nn_model.predict([self.X_test.AccountId, self._test.BookId, self.exogenous_valid])
        df_id = pd.DataFrame(np.array(self.X_test.AccountId))
        df_Book_id = pd.DataFrame(np.array(self.X_test.BookId))
        df_actual_rating = pd.DataFrame(np.array(self.y_test))
        df_preds = pd.DataFrame(preds)
        dfList = [df_id, df_Book_id, df_actual_rating, df_preds]  # List of your dataframes
        avp = pd.concat(dfList, ignore_index=True, axis=1)
        # new_df = pd.concat([new_df,df_preds],ignore_index=True,axis=1)
        avp.rename(columns={avp.columns[0]: "AccountId"}, inplace=True)
        avp.rename(columns={avp.columns[1]: "BookId"}, inplace=True)
        avp.rename(columns={avp.columns[2]: "Rate"}, inplace=True)
        avp.rename(columns={avp.columns[3]: "Pred_Rating"}, inplace=True)
        print(avp)

        print(avp['Pred_Rating'].max(), avp['Pred_Rating'].min())

        test_user_list = avp.AccountId.unique().tolist()
        overlap_summary = {}
        top_recos_to_check = 10
        for users in test_user_list:
            overlap_summary[users] = self.check_overlap(users, top_recos_to_check, avp)

        sorted_summary = sorted(overlap_summary.items(), key=lambda x: x[1], reverse=True)
        max_overlap = np.array(list(overlap_summary.values())).max()
        min_overlap = np.array(list(overlap_summary.values())).min()
        mean_overlap = np.array(list(overlap_summary.values())).mean()
        print("Max overlap in top" + str(top_recos_to_check) + " books " + str(max_overlap))
        print("Min overlap in top " + str(top_recos_to_check) + " books " + str(min_overlap))
        print("Average overlap in top " + str(top_recos_to_check) + " books " + str(mean_overlap))

    def test(self):
        pass

    @staticmethod
    def mix_max_scaler(df, scaling_cols):
        result = df.copy()
        for feature_name in scaling_cols:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    @staticmethod
    def check_overlap(UserId, top_recos_to_check, avp):
        samp_cust = avp[avp['AccountId'] == UserId][['AccountId', 'Rate', 'BookId']]
        samp_cust.sort_values(by='Rate', ascending=False, inplace=True)
        available_actual_ratings = samp_cust.shape[0]
        rows_to_fetch = min(available_actual_ratings, top_recos_to_check)
        preds_df_sampcust = avp[avp['AccountId'] == UserId][['AccountId', 'Pred_Rating', 'BookId']]
        preds_df_sampcust.sort_values(by='Pred_Rating', ascending=False, inplace=True)
        actual_rating = samp_cust.iloc[0:rows_to_fetch, :]
        pred_rating = preds_df_sampcust.iloc[0:rows_to_fetch, :]
        overlap = pd.Series(list(set(actual_rating.BookId).intersection(set(pred_rating.BookId))))
        pct_overlap = (len(overlap) / rows_to_fetch) * 100
        # print("Percentage of overlap in top"+str(top_recos_to_check)+" for User AccountId - "+str(UserId)+" : "+str(pct_overlap))
        return pct_overlap

    @staticmethod
    def build_model(dropout, latent_factors, n_books, n_users):
        n_latent_factors = latent_factors  # hyperparamter to deal with.
        user_input = Input(shape=(1,), name='user_input', dtype='int64')
        user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)
        user_vec = Dropout(dropout)(user_vec)
        book_input = Input(shape=(1,), name='book_input', dtype='int64')
        book_embedding = Embedding(n_books, n_latent_factors, name='book_embedding')(book_input)
        book_vec = Flatten(name='FlattenBooks')(book_embedding)
        book_vec = Dropout(dropout)(book_vec)
        sim = dot([user_vec, book_vec], name='Similarity-Dot-Product', axes=1)
        ###Exogenous Features input
        exog_input = Input(shape=(6,), name='exogenous_input', dtype='float64')
        exog_embedding = Embedding(6, 20, name='exog_embedding')(exog_input)
        # exog_embedding = Dense(65,activation='relu',name='exog_Dense')(exog_input)
        exog_vec = Flatten(name='FlattenExog')(exog_embedding)
        ##############
        nn_inp = Add(dtype='float64', name='Combine_inputs')([sim, exog_vec])
        nn_inp = Dense(128, activation='relu')(nn_inp)
        nn_inp = Dropout(dropout)(nn_inp)
        nn_inp = Dense(64, activation='relu')(nn_inp)
        nn_inp = BatchNormalization()(nn_inp)
        nn_output = Dense(1, activation='relu')(nn_inp)
        nn_model = Model([user_input, book_input, exog_input], nn_output)
        return nn_model
