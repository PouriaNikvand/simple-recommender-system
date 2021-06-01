import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.layers import Dropout, Flatten, Activation, Input, Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model

from managers.df_data import DfData
from configs.configs import Configs

""" Author: Pouria Nikvand """


class DeepModel:

    def __init__(self, my_df_data: DfData):
        self.my_df_data = my_df_data
        self.deep_model_configs = Configs.deep_model_configs
        self.train()

    def train(self):
        user_rating = self.preprocess(self.my_df_data.book_df, self.my_df_data.rating_df)
        self.users = user_rating.AccountId.unique()
        self.books = user_rating.BookId.unique()
        self.nn_model = self.build_model(self.deep_model_configs['dropout'],
                                         self.deep_model_configs['n_latent_factors'],
                                         len(self.books),
                                         len(self.users))

        self.user_rating, x_train, x_test, y_train, y_test, exogenous_train, exogenous_valid = self.BRS_pretrain_data(
            user_rating)

        history = self.BRS_train(x_train, x_test, y_train, y_test, exogenous_train, exogenous_valid)

        if not self.deep_model_configs['production_flag']:
            self.plot_training_results(history)
            self.result_analysis(x_test, y_test, exogenous_valid)

    @staticmethod
    def plot_training_results(history):
        plt.plot(history.history['loss'], 'g')
        plt.plot(history.history['val_loss'], 'b')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.show()

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

        scaling_cols = ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum', 'RaCount']

        book_rating_scaled = self.mix_max_scaler(book_rating, scaling_cols)
        book_rating_df = book_rating_scaled[
            ['BookId', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C', 'Sum', 'RaCount']]

        ##Let's create Book_id that we can use
        book_id = book_rating_df[['BookId']]

        user_rating = pd.merge(user_rating, book_id, on='BookId', how='left')
        book_rating_df = pd.merge(book_rating_df, book_id, on='BookId', how='left')

        book_rating_numeric = book_rating_df[['BookId', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']]
        user_rating = pd.merge(user_rating, book_rating_numeric, on='BookId', how='left')

        user_rating.fillna(0, inplace=True)

        return user_rating

    def BRS_pretrain_data(self, user_rating):
        userid2idx = {o: i for i, o in enumerate(self.users)}
        bookid2idx = {o: i for i, o in enumerate(self.books)}
        user_rating['AccountId_original'] = user_rating['AccountId']
        user_rating['BookId_original'] = user_rating['BookId']
        user_rating['AccountId'] = user_rating['AccountId'].apply(lambda x: userid2idx[x])
        user_rating['BookId'] = user_rating['BookId'].apply(lambda x: bookid2idx[x])
        self.y = user_rating['Rate']
        self.x = user_rating

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                            random_state=42)

        if self.deep_model_configs['production_flag']:
            x_train, y_train = self.x, self.y

        print(x_train.shape, x_test.shape)

        exogenous_train = np.array(x_train[
                                       ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']])
        exogenous_valid = np.array(x_test[
                                       ['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']])
        return user_rating, x_train, x_test, y_train, y_test, exogenous_train, exogenous_valid

    def BRS_train(self, x_train, x_test, y_train, y_test, exogenous_train, exogenous_valid):

        self.nn_model.compile(optimizer=Adam(lr=self.deep_model_configs['learning_rate']), loss='mse')

        history = self.nn_model.fit([x_train.AccountId, x_train.BookId, exogenous_train], y_train,
                                    batch_size=self.deep_model_configs['batch_size'],
                                    epochs=self.deep_model_configs['epochs'],
                                    validation_data=(
                                        [x_test.AccountId, x_test.BookId, exogenous_valid], y_test),
                                    verbose=1)

        return history

    def result_analysis(self, x_test, y_test, exogenous_valid):
        preds = self.nn_model.predict([x_test.AccountId, x_test.BookId, exogenous_valid])
        df_id = pd.DataFrame(np.array(x_test.AccountId))
        df_book_id = pd.DataFrame(np.array(x_test.BookId))
        df_actual_rating = pd.DataFrame(np.array(y_test))
        df_preds = pd.DataFrame(preds)
        df_list = [df_id, df_book_id, df_actual_rating, df_preds]  # List of your dataframes
        avp = pd.concat(df_list, ignore_index=True, axis=1)
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

    def test(self, user_id,num_recommendations):
        my_test = pd.DataFrame(self.user_rating['BookId'].unique(), columns=['BookId'])
        user_index_id = self.user_rating[self.user_rating['AccountId_original'] == user_id]['AccountId'].iloc[0]
        my_test['AccountId'] = user_index_id
        tmp = self.user_rating[['BookId', 'BookId_original', 'Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']]
        tmp = tmp.drop_duplicates()
        my_test = my_test.merge(tmp, on='BookId', how='left')
        exogenous_test = np.array(my_test[['Rate1C', 'Rate2C', 'Rate3C', 'Rate4C', 'Rate5C']])
        preds = self.nn_model.predict([my_test.AccountId, my_test.BookId, exogenous_test])
        print(preds)
        preds = pd.DataFrame(preds, columns=['Predictions'])
        result = preds.sort_values('Predictions', ascending=False)
        result = pd.DataFrame(result.index.values, columns=['BookId'])
        tmp = self.user_rating[['BookId', 'BookId_original']].drop_duplicates()
        result = result.merge(tmp, on='BookId', how='left')
        return pd.DataFrame(result['BookId_original'].values[:num_recommendations],columns=['BookId'])

    @staticmethod
    def mix_max_scaler(df, scaling_cols):
        result = df.copy()
        for feature_name in scaling_cols:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    @staticmethod
    def check_overlap(user_id, top_recos_to_check, avp):
        samp_cust = avp[avp['AccountId'] == user_id][['AccountId', 'Rate', 'BookId']]
        samp_cust.sort_values(by='Rate', ascending=False, inplace=True)
        available_actual_ratings = samp_cust.shape[0]
        rows_to_fetch = min(available_actual_ratings, top_recos_to_check)
        preds_df_sampcust = avp[avp['AccountId'] == user_id][['AccountId', 'Pred_Rating', 'BookId']]
        preds_df_sampcust.sort_values(by='Pred_Rating', ascending=False, inplace=True)
        actual_rating = samp_cust.iloc[0:rows_to_fetch, :]
        pred_rating = preds_df_sampcust.iloc[0:rows_to_fetch, :]
        overlap = pd.Series(list(set(actual_rating.BookId).intersection(set(pred_rating.BookId))))
        pct_overlap = (len(overlap) / rows_to_fetch) * 100
        # print("Percentage of overlap in top"+str(top_recos_to_check)+" for User AccountId - "+str(UserId)+" : "+str(pct_overlap))
        return pct_overlap

    @staticmethod
    def build_model(dropout, n_latent_factors, n_books, n_users):
        # hyperparamter to deal with.
        user_input = Input(shape=(1,), name='user_input', dtype='int64')
        user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)
        user_vec = Dropout(dropout)(user_vec)
        book_input = Input(shape=(1,), name='book_input', dtype='int64')
        book_embedding = Embedding(n_books, n_latent_factors, name='book_embedding')(book_input)
        book_vec = Flatten(name='FlattenBooks')(book_embedding)
        book_vec = Dropout(dropout)(book_vec)
        sim = dot([user_vec, book_vec], name='Similarity-Dot-Product', axes=1)
        # Exogenous Features input
        exog_input = Input(shape=(5,), name='exogenous_input', dtype='float64')
        exog_embedding = Embedding(5, 20, name='exog_embedding')(exog_input)
        # exog_embedding = Dense(65,activation='relu',name='exog_Dense')(exog_input)
        exog_vec = Flatten(name='FlattenExog')(exog_embedding)

        nn_inp = Add(dtype='float64', name='Combine_inputs')([sim, exog_vec])
        nn_inp = Dense(128, activation='relu')(nn_inp)
        nn_inp = Dropout(dropout)(nn_inp)
        nn_inp = Dense(64, activation='relu')(nn_inp)
        nn_inp = BatchNormalization()(nn_inp)
        nn_output = Dense(1, activation='relu')(nn_inp)
        nn_model = Model([user_input, book_input, exog_input], nn_output)

        nn_model.summary()
        return nn_model
