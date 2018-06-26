from __future__ import print_function, division
import pickle
import pandas as pd
import numpy as np
import random
from pprint import pprint

# loc_id 0-40558
# tim_id 0-23
# day_id 0-6
# week_id 0-3


def read_dict(_dict, k):
    for key in list(_dict.keys())[:k]:
        print(key)
        print(_dict[key])


def change_format():
    with open('../../../dataset/TRAJ.pkl', 'rb') as f:
        data = pickle.load(f)

    # friendship
    friendship = []
    for user1 in data.keys():
        for cate in data[user1]['friendship']:
            for user2 in data[user1]['friendship'][cate]:
                friendship.append((user1, user2) if user1 < user2 else (user2, user1))
    users = []
    for item in friendship:
        users.extend(list(item))
    users = set(users)

    # negative sampling
    friendship = set(friendship)
    negative = []
    for item in friendship:
        for i in range(2):
            candi = random.randint(0, max(users))
            negative.append((item[i], candi) if item[i] < candi else (candi, item[i]))
    negative = set(negative) - friendship
    negative = random.sample(negative, len(friendship))
    friendship = list(friendship)
    friendship = [list(item) + [1] for item in friendship]
    negative = [list(item) + [0] for item in negative]

    friendship_df = pd.DataFrame(friendship, columns=['user1', 'user2', 'friend'])
    negative_df = pd.DataFrame(negative, columns=['user1', 'user2', 'friend'])

    merge_df = pd.concat([friendship_df, negative_df], axis=0)
    merge_df.sort_values(by=['user1', 'friend']).to_csv('../../../dataset/friendship.csv', header=False, index=False)

    # trajectory
    traj_data = {}
    for key in data.keys():
        traj_data[key] = data[key]['traj']
    for user in traj_data.keys():
            for week in traj_data[user].keys():
                for day in traj_data[user][week].keys():
                    traj_data[user][week][day] = traj_data[user][week][day]

    with open('../../../dataset/traj_data.pkl', 'wb') as f:
        pickle.dump(traj_data, f)


def count():
    with open('../../../dataset/traj_data.pkl', 'rb') as f:
        traj_data = pickle.load(f)
    users = []
    weeks = []
    days = []
    locs = []
    for user in traj_data.keys():
        users.append(user)
        for week in traj_data[user].keys():
            weeks.append(week)
            for day in traj_data[user][week].keys():
                days.append(day)
                locs.append(traj_data[user][week][day][0])
    users = set(users)
    weeks = set(weeks)
    days = set(days)
    locs = set(locs)
    print(len(users), len(weeks), len(days), len(locs))
    print(1)

change_format()
