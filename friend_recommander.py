from surprise import KNNBasic
import pandas as pd
from surprise import Dataset
from surprise import Reader


movies = ['Star wars', 'Star wars', 'GOT', 'GOT', 'South Park', 'South Park', 'Harry potter', 'Harry potter']
rating = [1, 5, 1, 1, 5, 3, 2, 5]
users = ['Kim', 'Tim', 'John', 'Jimmy', 'Julia', 'Kim', 'Jimmy', 'Kim']

rating_dict = {
    'users': users,
    'items': movies,
    'rating': rating
}

def frRecommender(user):
    df = pd.DataFrame(rating_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['users', 'items', 'rating']], reader)
    trainset = data.build_full_trainset()
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    algo = KNNBasic(sim_options)
    algo.fit(trainset)

    uid = trainset.to_inner_uid(user)
    pred = algo.get_neighbors(uid, 3)
    for i in pred:
        print(trainset.to_raw_uid(i))

frRecommender('Kim')