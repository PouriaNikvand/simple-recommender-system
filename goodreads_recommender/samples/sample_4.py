# https://www.kaggle.com/sandy1112/book-reco-mf-using-svd-als-and-deep-learning

# **Credits
# http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
# https://github.com/jeffheaton/t81_558_deep_learning
# https://www.kaggle.com/rajmehra03/cf-based-recsys-by-low-rank-matrix-factorization
# https://beckernick.github.io/matrix-factorization-recommender/
# https://www.kaggle.com/vikashrajluhaniwal/matrix-factorization-recommendation-using-pyspark


import os
import pickle
from pathlib import PurePosixPath

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.image as mpimgimport
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from statistics import mean
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from random import shuffle
from zipfile import ZipFile
##Deep Learning specific stuff
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Flatten, Activation, Input, Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model

from pyspark import SparkContext, SQLContext  # required for dealing with dataframes
import numpy as np
from pyspark.ml.recommendation import ALS  # for Matrix Factorization using ALS


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def main():
    book_rating, user_rating = load_data()
    books, users, user_rating = preprocess(book_rating, user_rating)
    BRS(books, users, user_rating)


def load_data():
    path = str(PurePosixPath(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent)
    try:
        with open(path + '/dataset/goodreads-book-datasets-10m-sample3.pkl', 'rb') as file_in:
            book_rating = pickle.load(file_in)
    except FileNotFoundError:
        book_rating = pd.DataFrame()
        for file in glob.glob(path + "/dataset/goodreads-book-datasets-10m/book*.csv"):
            df = pd.read_csv(file)
            if book_rating.empty:
                book_rating = df
            else:
                book_rating.append(df, ignore_index=True)

        with open(path + '/dataset/goodreads-book-datasets-10m-sample3.pkl', 'wb') as file_out:
            pickle.dump(book_rating, file_out)

    try:
        with open(path + '/dataset/goodreads-book-datasets-10m-userrate-sample3.pkl', 'rb') as file_in:
            user_rating = pickle.load(file_in)
    except FileNotFoundError:
        user_rating = pd.DataFrame()
        for file in glob.glob(path + "/dataset/goodreads-book-datasets-10m/user_rating*.csv"):
            df = pd.read_csv(file)
            if user_rating.empty:
                user_rating = df
            else:
                user_rating.append(df, ignore_index=True)

        with open(path + '/dataset/goodreads-book-datasets-10m-userrate-sample3.pkl', 'wb') as file_out:
            pickle.dump(user_rating, file_out)

    print(book_rating.shape, user_rating.shape)
    return book_rating, user_rating


def preprocess(book_rating, user_rating):
    # Let's take a quick look at the two tables
    book_rating.head(3)
    user_rating.head()
    user_rating['Rating'].unique()

    # For the purpose of illustrating Collaborative filetring,
    # we can use user_rating as it is in the format where we have Users who have rated specific books.
    # We will skip the cases where there is no book name and no rating as well (as shown in the first few rows)

    print("Number of unique users in the user_rating table :" + str(user_rating['ID'].nunique()))

    # The ratings are given in sentences, let's convert them to numeric field on a scale of 0-5 (where 0=No rating)

    le = preprocessing.LabelEncoder()
    user_rating['Rating_numeric'] = le.fit_transform(user_rating.Rating.values)
    user_rating.tail()
    user_rating.head()

    # Creating Training and Validation set for the recommender system
    #
    # for this purpose, we first take only those customers whose ratings are > 0 i.e.
    # they have given some sort of rating and then take a random sample out of them i.e.
    # we hide few of the previously rated books as the test set. Let's see an example of how this is to be done.

    user_rating_pos = user_rating[user_rating['Rating_numeric'] > 0]
    user_rating_zero = user_rating[user_rating['Rating_numeric'] == 0]
    pos_rating_summary = user_rating_pos[['ID', 'Rating_numeric']].groupby(['ID']).agg(['count'])
    pos_rating_summary.columns = ['_'.join(col) for col in pos_rating_summary.columns.values]
    pos_rating_summary.reset_index(inplace=True)
    plt.figure(figsize=(8, 8))
    plt.title('Number of Ratings Density plot')
    sns.kdeplot(pos_rating_summary['Rating_numeric_count'], color="blue", shade=True)
    plt.show()

    # There seems to be a long tail, where some customers do provide a large number of ratings,
    # whereas bulk of the people seem to have more than 5 ratings at least.
    # Let's do a quick check to confirm this

    # We will choose our test train split from these set of users who have atleast 5 or more reviews.
    # This will ensure that if we take a 80-20 split, we have atleast 1 book that is held out for test predictions

    eligible_customer_list = pos_rating_summary[pos_rating_summary['Rating_numeric_count'] >= 5].ID.tolist()
    user_rating_eligible = user_rating_pos[user_rating_pos.ID.isin(eligible_customer_list)]
    user_rating_NotEligible = user_rating_pos[~user_rating_pos.ID.isin(eligible_customer_list)]
    print(user_rating_NotEligible.shape, user_rating_eligible.shape, user_rating_pos.shape)

    ##Create test and train split
    split_idx = np.random.rand(len(user_rating_eligible)) < 0.8
    user_rating_train_temp = user_rating_eligible[split_idx]
    user_rating_test = user_rating_eligible[~split_idx]
    user_rating_train = user_rating_train_temp.append(user_rating_NotEligible, ignore_index=True)
    print(user_rating_train.shape, user_rating_train_temp.shape, user_rating_NotEligible.shape)

    user_rating_train.head()

    # Data Pre-processing for Matrix Factorization
    # For using SVD to do Matrix Factorization we need to convert it into wide format from Long format.

    user_rating_wide = user_rating_train.pivot(index='ID', columns='Name', values='Rating_numeric').fillna(0)
    user_rating_wide.head()

    user_rating_matrix = user_rating_wide.to_numpy()
    user_ratings_mean = np.mean(user_rating_matrix, axis=1)
    user_rating_matrix_dm = user_rating_matrix - user_ratings_mean.reshape(-1, 1)
    ##Normalizing the ratings here, one can try a version without
    ##this and see if the recommendations are any better/worse

    ## k is a hyperparam here
    ## U and V are User latent matrix and V is Books latent matrix in our case
    U, sigma, V = svds(user_rating_matrix_dm, k=100)

    # Let's do matrix factorization using SVD now

    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), V) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_rating_wide.columns)

    test_user_list = user_rating_test.ID.unique().tolist()
    overlap_summary = {}
    top_recos_to_check = 10
    for users in test_user_list:
        if check_overlap(users, top_recos_to_check, user_rating_test, preds_df) is not None:
            overlap_summary[users] = check_overlap(users, top_recos_to_check, user_rating_test, preds_df)

    sorted_summary = sorted(overlap_summary.items(), key=lambda x: x[1], reverse=True)
    max_overlap = np.array(list(overlap_summary.values())).max()
    min_overlap = np.array(list(overlap_summary.values())).min()
    mean_overlap = np.array(list(overlap_summary.values())).mean()
    print("Max overlap in top" + str(top_recos_to_check) + " books " + str(max_overlap))
    print("Min overlap in top " + str(top_recos_to_check) + " books " + str(min_overlap))
    print("Average overlap in top " + str(top_recos_to_check) + " books " + str(mean_overlap))

    # Thus, on an average we are able to recommend the right books decent number of the times. This was set up with very little effort. Let's see if we can improve this further by using Alternating Least Squares.
    #
    # For this we will use Pyspark where there is a very good implementation of ALS
    #
    # Alternating Least Squares
    #
    # In any matrix factorization problem,
    # we are trying to find out a relatively small number k and
    # approximate each user u with a k dimensional vector xu and each book i with a k dimensional vector yi .
    # These vectors are referred as factors.
    # Then to predict user us rating for book i , we simply predict r~xu.Tyi .
    # In matrix notation it looks like as follows:

    # To estimate complete ratings, this is formulated as an optimization excercise (given below).
    # Here we minimize the least squared errors of the observed ratings (along with regularization term)

    # Notice that this objective function is non-convex and is NP-hard to optimize.
    # Gradient descent can be used as an approximate approach here,
    # however it turns out to be slow and costs lots of iterations.
    # Note however, that if we fix the set of variables X and treat them as constants,
    # then the objective is a convex function of Y and vice versa.
    # In ALS, we fix Y and optimize X, then fix X and optimize Y ,
    # and repeat until convergence (algo shown below in picture)

    sc = SparkContext()  # instantiating spark context
    sqlContext = SQLContext(sc)  # instantiating SQL context

    # For the ALS implementation in pyspark, user id and book ids need to be in Integer format.
    # Let's create book id and create the required dataframe

    book_names_train = user_rating_train[['Name']]
    book_names_test = user_rating_test[['Name']]
    book_name = pd.concat([book_names_train, book_names_test], axis=0)
    book_name.drop_duplicates(inplace=True)
    book_name['Book_id'] = book_name.index.values
    book_name.head()

    user_rating_train = pd.merge(user_rating_train, book_name, on='Name', how='left')
    user_rating_test = pd.merge(user_rating_test, book_name, on='Name', how='left')
    user_rating_train.head()

    user_rating_train_ = user_rating_train[['ID', 'Book_id', 'Rating_numeric']]
    user_rating_test_ = user_rating_test[['ID', 'Book_id', 'Rating_numeric']]
    user_rating_train_.head()

    user_rating_train_.to_csv('user_rating_train.csv', index=False)
    user_rating_test_.to_csv('user_rating_test.csv', index=False)
    user_rating_train_sp = sqlContext.read.csv('user_rating_train.csv', header=True, inferSchema=True)
    user_rating_test_sp = sqlContext.read.csv('user_rating_test.csv', header=True, inferSchema=True)

    als = ALS(userCol="ID", itemCol="Book_id", ratingCol="Rating_numeric", rank=20, maxIter=10, seed=0, )
    model = als.fit(user_rating_train_sp)

    model.userFactors.show(5, truncate=False)  # displaying the latent features for five user

    predictions = model.transform(user_rating_test_sp[["ID", "Book_id"]])

    ratesAndPreds = user_rating_test_sp.join(other=predictions, on=['ID', 'Book_id'], how='inner').na.drop()
    ratesAndPreds.show(5)

    # converting the columns into numpy arrays for direct and easy calculations
    rating = np.array(ratesAndPreds.select("Rating_numeric").collect()).ravel()
    prediction = np.array(ratesAndPreds.select("prediction").collect()).ravel()
    print("RMSE : ", np.sqrt(np.mean((rating - prediction) ** 2)))

    avp_als = ratesAndPreds.toPandas()
    print(avp_als.head())

    test_user_list = avp_als.ID.unique().tolist()
    overlap_summary = {}
    top_recos_to_check = 10
    for users in test_user_list:
        overlap_summary[users] = check_overlap_2(users, top_recos_to_check, avp_als)

    sorted_summary = sorted(overlap_summary.items(), key=lambda x: x[1], reverse=True)
    max_overlap = np.array(list(overlap_summary.values())).max()
    min_overlap = np.array(list(overlap_summary.values())).min()
    mean_overlap = np.array(list(overlap_summary.values())).mean()
    print("Max overlap in top" + str(top_recos_to_check) + " books " + str(max_overlap))
    print("Min overlap in top " + str(top_recos_to_check) + " books " + str(min_overlap))
    print("Average overlap in top " + str(top_recos_to_check) + " books " + str(mean_overlap))

    # book_map = user_rating_temp[['Name']]
    # book_map.drop_duplicates(subset=['Name'], keep='first', inplace=True)
    # book_map['Book_Id'] = book_map.index.values
    # user_rating_temp = pd.merge(user_rating_temp, book_map, on='Name', how='left')
    # user_rating = user_rating_temp[user_rating_temp['Name'] != 'Rating']  ##Dropping users who have not rated any books
    # user_rating.head()

    le = preprocessing.LabelEncoder()
    user_rating['Rating_numeric'] = le.fit_transform(user_rating.Rating.values)
    users = user_rating.ID.unique()
    books = user_rating.Book_Id.unique()
    return books, users, user_rating


def BRS(books, users, user_rating):
    model = Sequential()
    embedding_layer = Embedding(input_dim=10, output_dim=4, input_length=2)
    model.add(embedding_layer)
    model.compile('adam', 'mse')
    model.summary()

    input_data = np.array([[1, 2]])

    pred = model.predict(input_data)

    print(input_data.shape)
    print(pred)

    embedding_layer.get_weights()

    ##Train-test split  - keeping 80%-20% for simplicity. But one can create a k-fold set up for better accuracy as well
    userid2idx = {o: i for i, o in enumerate(users)}
    bookid2idx = {o: i for i, o in enumerate(books)}
    user_rating['ID'] = user_rating['ID'].apply(lambda x: userid2idx[x])
    user_rating['Book_Id'] = user_rating['Book_Id'].apply(lambda x: bookid2idx[x])
    y = user_rating['Rating_numeric']
    X = user_rating.drop(['Rating_numeric'], axis=1)
    ####
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape)

    n_books = len(user_rating['Book_Id'].unique())
    n_users = len(user_rating['ID'].unique())

    nn_model = build_model(0.4, 65, n_books, n_users)
    nn_model.summary()

    nn_model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    batch_size = 128
    epochs = 5
    History = nn_model.fit([X_train.ID, X_train.Book_Id], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([X_test.ID, X_test.Book_Id], y_test),
                           verbose=1)

    preds = nn_model.predict([X_test.ID, X_test.Book_Id])
    avp = (preds, y_test)
    df_id = pd.DataFrame(np.array(X_test.ID))
    df_Book_id = pd.DataFrame(np.array(X_test.Book_Id))
    df_actual_rating = pd.DataFrame(np.array(y_test))
    df_preds = pd.DataFrame(preds)
    dfList = [df_id, df_Book_id, df_actual_rating, df_preds]  # List of your dataframes
    avp = pd.concat(dfList, ignore_index=True, axis=1)
    # new_df = pd.concat([new_df,df_preds],ignore_index=True,axis=1)
    avp.rename(columns={avp.columns[0]: "ID"}, inplace=True)
    avp.rename(columns={avp.columns[1]: "Book_Id"}, inplace=True)
    avp.rename(columns={avp.columns[2]: "Rating_numeric"}, inplace=True)
    avp.rename(columns={avp.columns[3]: "Pred_Rating"}, inplace=True)
    print(avp)

    test_user_list = avp.ID.unique().tolist()
    overlap_summary = {}
    top_recos_to_check = 10
    for users in test_user_list:
        overlap_summary[users] = check_overlap_3(users, top_recos_to_check, avp)

    sorted_summary = sorted(overlap_summary.items(), key=lambda x: x[1], reverse=True)
    max_overlap = np.array(list(overlap_summary.values())).max()
    min_overlap = np.array(list(overlap_summary.values())).min()
    mean_overlap = np.array(list(overlap_summary.values())).mean()
    print("Max overlap in top" + str(top_recos_to_check) + " books " + str(max_overlap))
    print("Min overlap in top " + str(top_recos_to_check) + " books " + str(min_overlap))
    print("Average overlap in top " + str(top_recos_to_check) + " books " + str(mean_overlap))

    # That's a huge improvement in the accuracy of recommendations by just switching
    # to a different way of doing low rank matrix factorization using a deep neural network.
    # Now we are able to recommend almost 3 out of 4 books correctly to the user.
    #
    # Further improvments can be made by playing around with network architecture.


def build_model(dropout, latent_factors, n_books, n_users):
    n_latent_factors = latent_factors  # hyperparamter to deal with.
    user_input = Input(shape=(1,), name='user_input', dtype='int64')
    user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding',
                               embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=42))(user_input)
    user_vec = Flatten(name='FlattenUsers')(user_embedding)
    user_vec = Dropout(dropout)(user_vec)
    book_input = Input(shape=(1,), name='book_input', dtype='int64')
    book_embedding = Embedding(n_books, n_latent_factors, name='book_embedding',
                               embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=42))(book_input)
    book_vec = Flatten(name='FlattenBooks')(book_embedding)
    book_vec = Dropout(dropout)(book_vec)
    sim = dot([user_vec, book_vec], name='Similarity-Dot-Product', axes=1)
    nn_inp = Dense(128, activation='relu')(sim)
    nn_inp = Dropout(dropout)(nn_inp)
    nn_inp = Dense(64, activation='relu')(nn_inp)
    nn_inp = BatchNormalization()(nn_inp)
    nn_inp = Dense(1, activation='relu')(nn_inp)
    nn_model = Model([user_input, book_input], nn_inp)
    return nn_model


##Function for checking overlap for one specific user
def check_overlap(UserId, top_recos_to_check, user_rating_test, preds_df):
    samp_cust = user_rating_test[user_rating_test['ID'] == UserId]
    samp_cust.sort_values(by='Rating_numeric', ascending=False, inplace=True)
    book_name_testcust = samp_cust.Name.unique().tolist()
    available_actual_ratings = samp_cust.shape[0]
    rows_to_fetch = min(available_actual_ratings, top_recos_to_check)
    preds_df_sampcust = preds_df[preds_df['ID'] == UserId]
    if preds_df_sampcust.shape[0] == 0:
        pass
    elif preds_df_sampcust.shape[0] > 0:
        preds_check_cust = preds_df_sampcust.T
        preds_check_cust.reset_index(inplace=True)
        preds_check_cust.rename(columns={preds_check_cust.columns[0]: "Name"}, inplace=True)
        # preds_check_cust = preds_df_sampcust_T[preds_df_sampcust_T['Name']!='ID']
        preds_check_cust.rename(columns={preds_check_cust.columns[1]: "Ratings_normalized_predicted"}, inplace=True)
        preds_check_cust.sort_values(by='Ratings_normalized_predicted', ascending=0, inplace=True)
        preds_check_cust_check = preds_check_cust[preds_check_cust.Name.isin(book_name_testcust)]
        actual_rating = samp_cust.iloc[0:rows_to_fetch, :]
        pred_rating = preds_check_cust_check.iloc[0:rows_to_fetch, :]
        overlap = pd.Series(list(set(actual_rating.Name).intersection(set(pred_rating.Name))))
        pct_overlap = (len(overlap) / rows_to_fetch) * 100
        # print("Percentage of overlap in top"+str(top_recos_to_check)+" for User ID - "+str(UserId)+" : "+str(pct_overlap))
        return pct_overlap
    return None

def check_overlap_2(UserId, top_recos_to_check, avp_als):
    samp_cust = avp_als[avp_als['ID'] == UserId][['ID', 'Rating_numeric', 'Book_id']]
    samp_cust.sort_values(by='Rating_numeric', ascending=False, inplace=True)
    available_actual_ratings = samp_cust.shape[0]
    rows_to_fetch = min(available_actual_ratings, top_recos_to_check)
    preds_df_sampcust = avp_als[avp_als['ID'] == UserId][['ID', 'prediction', 'Book_id']]
    preds_df_sampcust.sort_values(by='prediction', ascending=False, inplace=True)
    actual_rating = samp_cust.iloc[0:rows_to_fetch, :]
    pred_rating = preds_df_sampcust.iloc[0:rows_to_fetch, :]
    overlap = pd.Series(list(set(actual_rating.Book_id).intersection(set(pred_rating.Book_id))))
    pct_overlap = (len(overlap) / rows_to_fetch) * 100
    # print("Percentage of overlap in top"+str(top_recos_to_check)+" for User ID - "+str(UserId)+" : "+str(pct_overlap))
    return pct_overlap


def check_overlap_3(UserId, top_recos_to_check, avp):
    samp_cust = avp[avp['ID'] == UserId][['ID', 'Rating_numeric', 'Book_Id']]
    samp_cust.sort_values(by='Rating_numeric', ascending=False, inplace=True)
    available_actual_ratings = samp_cust.shape[0]
    rows_to_fetch = min(available_actual_ratings, top_recos_to_check)
    preds_df_sampcust = avp[avp['ID'] == UserId][['ID', 'Pred_Rating', 'Book_Id']]
    preds_df_sampcust.sort_values(by='Pred_Rating', ascending=False, inplace=True)
    actual_rating = samp_cust.iloc[0:rows_to_fetch, :]
    pred_rating = preds_df_sampcust.iloc[0:rows_to_fetch, :]
    overlap = pd.Series(list(set(actual_rating.Book_Id).intersection(set(pred_rating.Book_Id))))
    pct_overlap = (len(overlap) / rows_to_fetch) * 100
    # print("Percentage of overlap in top"+str(top_recos_to_check)+" for User ID - "+str(UserId)+" : "+str(pct_overlap))
    return pct_overlap


if __name__ == '__main__':
    main()
