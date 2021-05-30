from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import os
import glob
import pickle


def main():
    books, ratings = load_data()
    preprocess(books, ratings)
    BBY(books)

def load_data():
    path = str(PurePosixPath(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent)
    try:
        with open(path + '/dataset/goodreads-book-datasets-10m-sample2.pkl', 'rb') as file_in:
            books = pickle.load(file_in)
    except:
        list_of_files = os.listdir(path + '/dataset/goodreads-book-datasets-10m')
        all_data = []
        for each_file in list_of_files:
            if each_file.startswith('book'):
                print(each_file)
                df = pd.read_csv(path + '/dataset/goodreads-book-datasets-10m/' + each_file,
                                 usecols=['Name', 'Rating', 'PublishYear', 'Authors'])
                all_data.append(df)

        books = pd.concat(all_data, axis=0)
        with open(path + '/dataset/goodreads-book-datasets-10m-sample2.pkl', 'wb') as file_out:
            pickle.dump(books, file_out)

    try:
        with open(path + '/dataset/goodreads-book-datasets-10m-userrate-sample2.pkl', 'rb') as file_in:
            ratings = pickle.load(file_in)
    except:
        list_of_files = os.listdir(path + '/dataset/goodreads-book-datasets-10m')
        all_data = []
        for each_file in list_of_files:
            if each_file.startswith('user'):
                print(each_file)
                df = pd.read_csv(path + '/dataset/goodreads-book-datasets-10m/' + each_file,
                                 usecols=['Name', 'Rating'])
                all_data.append(df)

        ratings = pd.concat(all_data, axis=0)
        with open(path + '/dataset/goodreads-book-datasets-10m-userrate-sample2.pkl', 'wb') as file_out:
            pickle.dump(ratings, file_out)

    return books, ratings


def preprocess(data, data2):
    data.isna().any()
    data.sort_values('PublishYear', ascending=False).head()

    data2.isna().any()
    data_merge = pd.merge(data, data2, on = 'Name', how = 'right')
    data_merge.isna().any()
    data_merge.dropna(inplace=True)
    data_merge.isna().any()
    data_merge.head()

    data_merge.drop_duplicates(subset=['Name'], inplace=True)
    data_merge.duplicated().any()
    print(data_merge.PublishYear.max())
    print(data_merge.PublishYear.min())
    # limit the range of data from Year 1990 to 2020
    new_data = data_merge[data_merge['PublishYear'] >= 1990]
    new_data.head()
    new_data.Rating_y.unique()
    new_data['Rating_new'] = np.where(
        (new_data['Rating_y'] == 'it was amazing') | (new_data['Rating_y'] == 'really liked it'),
        float(4.5), np.where(new_data['Rating_y'] == 'liked it', float(3.8),
                             np.where(new_data['Rating_y'] == 'it was ok', float(3.5), float(2.0))))

    new_data.head()
    new_data.info()

    new_data['Rating_mean'] = ((new_data['Rating_x'] + new_data['Rating_new']) / 2).round(2).astype(float)
    new_data.head()
    new_data.Rating_mean.max()
    new_data.drop(['Rating_x', 'Rating_y', 'Rating_new'], axis=1, inplace=True)
    new_data = new_data.sort_values(by = ['PublishYear', 'Rating_mean'], ascending = [False, False])
    new_data.head()
    l = []
    for year in range(1990, 2021):
        new_data1 = new_data[new_data.PublishYear == year].iloc[0]
        #print(new_data1)
        l.append(new_data1)
        new_data2 = pd.DataFrame(l, columns = ['Name', 'PublishYear', 'Authors', 'Rating_mean'])
    new_data2 = new_data2.reset_index()
    new_data2.drop('index', axis=1, inplace=True)
    new_data2.PublishYear = new_data2.PublishYear.astype(int)



def BBY(books):
    pass

if __name__ == '__main__':
    main()
