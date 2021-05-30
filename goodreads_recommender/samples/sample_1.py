import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import PurePosixPath
from wordcloud import WordCloud, STOPWORDS

""" Author: Pouria Nikvand """


def main():
    books = load_data()
    books = preprocess(books)
    EDA(books)


def load_data():
    path = str(PurePosixPath(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent)
    try:
        with open(path + '/dataset/goodreads-book-datasets-10m-sample1.pkl', 'rb') as file_in:
            books = pickle.load(file_in)
    except FileNotFoundError:
        books = pd.DataFrame(
            columns=pd.read_csv(path + '/dataset/goodreads-book-datasets-10m/book1000k-1100k.csv',
                                index_col='Id').columns)
        for dirname, _, filenames in os.walk(path + '/dataset/goodreads-book-datasets-10m/'):
            for filename in filenames:
                if "_" not in filename:
                    books = pd.concat([books, pd.read_csv(os.path.join(dirname, filename), index_col='Id')])
                    print(os.path.join(dirname, filename), 'OK')

        with open(path + '/dataset/goodreads-book-datasets-10m-sample1.pkl', 'wb') as file_out:
            pickle.dump(books, file_out)
    books.sort_index(inplace=True)
    tmp_looking_data = books.head(3)
    print(books.info())

    # There are a lot of numerical data, that was interpreted as strigs, which I'm going to fix.

    # Rating columns (RatingDist5, RatingDist4, RatingDist3, RatingDist2, RatingDist1, RatingDistTotal)
    # start with redundant part like '5:', '4:', etc.
    # This information need to be checked and removed, if it is not needed.

    # PublishMonth = 16 - looks strange.

    # There are also a lot of missing values in Language, Description and Count of text reviews columns.

    return books


def preprocess(books):
    print("len books names : ", len(books['Name']))
    print("number of unique book names : ", books['Name'].nunique())
    # 214K duplicated names

    print("books with the same name, but from different Publishers: ",
          books[books['Name'].duplicated(keep=False)].sort_values('Name'))
    # books, that were published with the same name, but from different Publishers.

    # check 100% duplicates.
    print("number of 100٪ duplicated boooks : ", len(books[books.duplicated(keep=False)]))

    # Droping duplicates
    books.drop_duplicates(inplace=True)
    print('number of fully unique books : ', len(books))

    # after removing fully duplicated books check for remove same name author
    plt.figure(figsize=(12, 6))
    popular_names = sns.barplot(books[~books[['Name', 'Authors']].duplicated()]['Name'].value_counts().head(20).index,
                                books[~books[['Name', 'Authors']].duplicated()]['Name'].value_counts().head(20).values)
    popular_names.set_xticklabels(popular_names.get_xticklabels(), rotation=90)
    popular_names.set_xlabel('Book name')
    popular_names.set_ylabel('Number of books')
    plt.savefig('name-author-20top')
    plt.show()

    # Checking number of unique authors is dataset
    print('number of unique authors : ', books['Authors'].nunique())
    print('most books grouped by author : ')
    print(books.groupby('Authors')['Name'].count().sort_values(ascending=False).head(20))

    # Let's take a look on missing values, maybe those books are problematic and can be removed
    print('missing values in ISBN : ', len(books[books['ISBN'].isnull()]))

    # ratings
    print('discribe of ratings in books : ', books['Rating'].describe())

    sns.distplot(books['Rating'], bins=15, kde=False)
    plt.savefig('ratings')
    plt.show()

    # convert date's to numerical data.
    books['PublishYear'] = books['PublishYear'].astype('int')
    books['PublishMonth'] = books['PublishMonth'].astype('int')
    books['PublishDay'] = books['PublishDay'].astype('int')

    # Looking for descriptive statistics
    print('discribe the publish dates : ', books[['PublishYear', 'PublishMonth', 'PublishDay']].describe())
    # Minimal and maximal Years look quite strange. I need to investigate that.
    # Maximal Day is 12, but maximal Month is 31, so it is obvious, that data was mislabeled. Need to be fixed.

    # Replacing day and month
    books['PublishMonth'], books['PublishDay'] = books['PublishDay'], books['PublishMonth']

    # Looking into years
    print(books['PublishYear'].unique())

    print(books[(books['PublishYear'] < 1400) | (books['PublishYear'] > 2020)]['Name'].count())

    # In details
    print(books[(books['PublishYear'] < 1400) | (books['PublishYear'] > 2020)])

    # Removing books with errors in years
    books.drop((books[(books['PublishYear'] < 1800) | (books['PublishYear'] > 2020)].index).tolist(), inplace=True)

    # Checking missing values in Publisher column
    print(books[books['Publisher'].isnull()].head(10))

    # Also books with good ratings. I cannot remove it.

    # How many unique published are there?
    print(books['Publisher'].nunique())

    # Which publisher issued the biggest variety of books?
    print(books['Publisher'].value_counts().head(10))

    print(books.head(3))
    # get rid of that redundant part like '5:', '4:', etc.
    books['RatingDistTotal'] = books['RatingDistTotal'].apply(lambda rating: rating.split(':')[1]).astype('int')
    books['RatingDist1'] = books['RatingDist1'].apply(lambda rating: rating.split(':')[1]).astype('int')
    books['RatingDist2'] = books['RatingDist2'].apply(lambda rating: rating.split(':')[1]).astype('int')
    books['RatingDist3'] = books['RatingDist3'].apply(lambda rating: rating.split(':')[1]).astype('int')
    books['RatingDist4'] = books['RatingDist4'].apply(lambda rating: rating.split(':')[1]).astype('int')
    books['RatingDist5'] = books['RatingDist5'].apply(lambda rating: rating.split(':')[1]).astype('int')

    print(books[['RatingDistTotal', 'RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4',
                 'RatingDist5']].describe())

    # Changing data type
    books['CountsOfReview'] = books['CountsOfReview'].astype('int')

    print(books['CountsOfReview'].describe())

    print(books['CountsOfReview'].value_counts())

    # And let's check Count of text reviews right away
    books['Count of text reviews'] = books['Count of text reviews'].astype('float')
    print(books['Count of text reviews'].describe())

    print(books['Language'].unique())

    books['Language'] = books['Language'].str.replace('en-US', 'eng').str.replace('en-GB', 'eng').str.replace('en-CA',
                                                                                                              'eng').str.replace(
        'nl', 'nld')

    print(books[books['Language'] == '--'])
    # there was only 13 books need to review but no time then remove
    books.loc[books[books['Language'] == '--'].index, 'Language'] = np.nan

    # look on languages distribution
    print(books['Language'].value_counts())

    plt.figure(figsize=(12, 6))
    langs = sns.barplot(x=books['Language'].value_counts().head(5), y=books['Language'].value_counts().head(5).index)
    langs.set_xlabel('Number of books')
    langs.set_ylabel('Language')
    plt.savefig('language distribution')
    plt.show()

    # Changing data type
    books['pagesNumber'] = books['pagesNumber'].astype('int')

    print(books['pagesNumber'].describe())

    print(books[books['pagesNumber'] > 100000])

    books.drop((books[books['pagesNumber'] > 100000].index).tolist(), inplace=True)

    print(books['Description'])

    return books


def EDA(books):
    # Checking information again
    books.info()
    # Let's check the book with biggest number of rates (total)
    print(books[books['RatingDistTotal'] == books['RatingDistTotal'].max()])

    # And let's check the book with biggest number of 5-star rates
    print(books[books['RatingDist5'] == books['RatingDist5'].max()])

    print(books[books['Rating'] == 5])

    print(books[(books['Rating'] == 5) & (books['RatingDistTotal'] > 1000)])

    print(
        books[(books['Rating'] > 4.5) & (books['RatingDistTotal'] > 1000)].sort_values('Rating', ascending=False).head(
            3))
    # Let's check authors with biggest number of rates (total number for all books)
    print(books.groupby('Authors')['RatingDistTotal'].sum().sort_values(ascending=False).head(5))

    print(books.groupby('Authors')['RatingDist5'].sum().sort_values(ascending=False).head(5))

    print(books.groupby('Authors')['Name'].count().sort_values(ascending=False).head(10))

    print(books[
              ['RatingDistTotal', 'RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5',
               'CountsOfReview',
               'pagesNumber']].corr())

    plt.figure(figsize=(12, 6))
    books_years = sns.barplot(y=books.groupby(['PublishYear'])['Name'].count().tail(60),
                              x=books.groupby(['PublishYear'])['Name'].count().tail(60).index)
    books_years.set_xticklabels(books_years.get_xticklabels(), rotation=90)
    books_years.set_xlabel('Publish Year')
    books_years.set_ylabel('Number of books')
    plt.savefig('number of books per year')
    plt.show()

    print(books.groupby(['PublishYear'])['pagesNumber'].mean().tail(10))

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='PublishYear', y='pagesNumber', data=books)
    plt.savefig('page number per year line plot')
    plt.show()

    # Setting stopwords for names
    stopwords_names = set(STOPWORDS)
    stopwords_names.update(['book', 'story'])

    # Creating words list for names
    words_from_names = [word for rows in books['Name'].str.lower().str.split() for word in rows if
                        word not in stopwords_names]
    names = " ".join(name for name in words_from_names)

    # Creating a cloud with words from names:
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(max_words=30, background_color="white", colormap='copper').generate(names)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Creating words list from descriptions
    words_from_description = [word for rows in
                              books['Description'].dropna().str.replace('/', '').str.replace('\\', '').str.replace(
                                  '<br>', '').str.replace('<p>', '').str.replace('><br', '').str.replace('<br',
                                                                                                         '').str.replace(
                                  '<', '').str.replace('>', '').str.replace('--', '').str.replace('.', '').str.replace(
                                  ',', '').str.lower().str.split() for word in rows if word not in STOPWORDS]

    # Creating a cloud with words from descriptions taken top 200 words:
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(max_words=60, background_color="white", colormap='copper').generate_from_frequencies(
        frequencies=pd.Series(words_from_description).value_counts().head(100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
