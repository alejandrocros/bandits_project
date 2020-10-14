import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def update(historical_dataset, matrix, t, batch_size, recommender):
    # reward if rec matches logged data, ignore otherwise
    actions = matrix[t:t+batch_size]
    actions = actions.loc[actions['Items'].isin(recommender)]
    # add row to history if recs match logging policy
    historical_dataset = historical_dataset.append(actions)
    results = actions[['Items', 'Ratings']]

    return historical_dataset, results


def epsilon_greedy_strategy(historical, arms, cluster_size, epsilon):

    # Binomial distribution (0,1) with a probability epsilon to draw 1, choosing to explore another arm at random
    exploration_or_exploitation = np.random.binomial(1,epsilon)

    if exploration_or_exploitation==1 or historical.shape[0]==0:
        # Explore a new arm at random
        recommender = np.random.choice(arms, size = (cluster_size), replace = False)
    else:
        # Exploitation
        exploitation = historical[['Items', 'Ratings']].groupby('Items').agg({'Ratings': ['mean', 'count']})
        exploitation.columns = ['Reward', 'Users_number']
        exploitation['Items'] = exploitation.index
        exploitation= exploitation.sort_values('Reward', ascending=False)
        recommender = exploitation.loc[exploitation.index[0:cluster_size], 'Items'].values

    return recommender


def ucb_strategy(historical, t, cluster_size, arms, bayesian, ucb_confidence):

    if historical.shape[0]==0:
        recommender = np.random.choice(arms, size=(cluster_size), replace=False)
    else:
        scores = historical[['Items', 'Ratings']].groupby('Items').agg({'Ratings': ['mean', 'count', 'std']})
        scores.columns = ['mean', 'count', 'vol']

        if bayesian==True:
            scores['UCB'] = scores['mean'] + ucb_confidence * scores['vol'] / scores['count']
        else:
            scores['UCB'] = scores['mean'] + np.sqrt(((2 * np.log10(t)) / scores['count']))

        scores['Items'] = scores.index
        scores = scores.sort_values('UCB', ascending=False)
        recommender = scores.loc[scores.index[0:cluster_size], 'Items'].values

    return recommender


def main():

    # Preprocessing
    users = pd.read_csv('U_movielens.csv',header=None) # Users
    arms = pd.read_csv('Vt_movielens.csv', header=None) # Items/movies (arms)
    ratings_hat = np.dot(users,arms) # (users*items)
    print(ratings_hat.shape)

    nb_users = ratings_hat.shape[0]
    nb_items =  ratings_hat.shape[1]
    total_users_items = ratings_hat.shape[0] * ratings_hat.shape[1]

    # 1 is the film have been appreciated, 0 otherwise
    ratings_hat = np.where(ratings_hat < 4.0, 0, ratings_hat)
    ratings_hat = np.where(ratings_hat >= 4.0, 1, ratings_hat)

    #Time series and random.shuffle(ratings_hat)
    matrix_reordered = pd.DataFrame([0] *  nb_items)
    matrix_items = pd.DataFrame(list(range(0,nb_items))*nb_users)

    for i in range(1,nb_users):
        matrix_reordered = matrix_reordered.append(pd.DataFrame([i]*nb_items))

    matrix_reordered  = matrix_reordered.rename(columns={0: 'Users'})
    matrix_reordered['Items'] = matrix_items.values
    matrix_reordered['Ratings'] = ratings_hat.reshape(ratings_hat.shape[0] * ratings_hat.shape[1])
    matrix_reordered.index = range(0,total_users_items)
    matrix_reordered['TimeStamps'] = matrix_reordered.index

    matrix_shuffled = matrix_reordered.sample(frac=1)
    matrix_shuffled.index = range(0,total_users_items)
    matrix_shuffled['TimeStamps'] = matrix_shuffled.index
    #### TO BE CHANGED
    matrix_shuffled = matrix_shuffled[0:100000]
    #### TO BE CHANGED

    # Strategy implementation
    historical_dataset = pd.DataFrame(data=None, columns=matrix_shuffled.columns)
    cluster_size = 1
    batch_size = 10

    rewards = []

    for t in range(matrix_shuffled.shape[0] // batch_size):
        # T steps
        t = t * batch_size

        # Recommender
        #recommender = epsilon_greedy_strategy(historical=historical_dataset.loc[historical_dataset.TimeStamps <= t,], arms=matrix_shuffled.Items.unique(), cluster_size=cluster_size, epsilon=0.2)
        recommender = ucb_strategy(historical=historical_dataset.loc[historical_dataset.TimeStamps <= t,], t = t, cluster_size = cluster_size, arms=matrix_shuffled.Items.unique(),bayesian = False, ucb_confidence=1.5)

        # Update
        historical_dataset, score = update(historical_dataset, matrix_shuffled, t, batch_size, recommender)

        if score is not None:
            score = score.Ratings.tolist()
            rewards.extend(score)

