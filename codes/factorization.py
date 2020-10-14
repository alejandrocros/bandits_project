import numpy as np
import pandas as pd
# from scipy.linalg import clarkson_woodruff_transform
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

def main():

    ## Movielens-1M
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::',
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding='latin1',
                            engine='python')
    movies = pd.read_table('ml-1m/movies.dat', sep='::',
                           names=['MovieID', 'Title', 'Genres'],
                           encoding='latin1',
                           engine='python')
    users = pd.read_table('ml-1m/users.dat', sep='::',
                          names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip'],
                          encoding='latin1',
                          engine='python')

    ## Films with sufficient numbers of reviews
    N = 1000
    ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
    # top_ratings = ratings_count.sort_values(ascending=False)[:N]
    top_ratings = ratings_count[ratings_count >= N]
    top_ratings.head(10)

    # movies_topN = movies[movies.MovieID.isin(top_ratings.index)]
    # print('Shape: {}'.format(movies_topN.shape))
    # movies_topN
    ratings_topN = ratings[ratings.MovieID.isin(top_ratings.index)]
    print('Shape: {}'.format(ratings_topN.shape))
    ratings_topN.head(10)

    n_users = ratings_topN.UserID.unique().shape[0]
    n_movies = ratings_topN.MovieID.unique().shape[0]
    print('Number of users = {} | Number of movies = {}'.format(n_users, n_movies))

    ## Low Rank Matrix Factorization
    R_df = ratings_topN.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    R_df.head()

    M = R_df.values
    sparsity = round(1.0 - np.count_nonzero(M) / float(n_users * n_movies), 3)
    print('Number of users = {} | Number of movies = {}'.format(n_users, n_movies))
    print('The sparsity level is {}%'.format(sparsity * 100))

    K = 30 #numbers of topics

    # Sparse SVD
    U, s, Vt = svds(M, k=K)
    s = np.diag(s)
    U = np.dot(U, s)
    print('U: {}'.format(U.shape))
    print('Vt: {}'.format(Vt.shape))

    np.savetxt('U_movielens.csv', U, delimiter=',')
    np.savetxt('Vt_movielens.csv', Vt, delimiter=',')

    # Non-negative matrix factorization (NMF)

    model = NMF(n_components=K, init='random', random_state=0)
    W = model.fit_transform(M)
    H = model.components_
    print('W: {}'.format(W.shape))
    print('H: {}'.format(H.shape))

    np.savetxt('U_movielens.csv', W, delimiter=',')
    np.savetxt('Vt_movielens.csv', H, delimiter=',')


