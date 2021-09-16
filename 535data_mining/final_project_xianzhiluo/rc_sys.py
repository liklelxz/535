import sklearn
import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from surprise import\
    Reader, Dataset, accuracy
from surprise import\
    SVD, SVDpp, KNNWithMeans
from surprise.model_selection import\
    train_test_split,GridSearchCV,cross_validate


def build_matrix(inputf,outputf):
    a = np.zeros((943, 1682))
    with open(inputf) as f:
        for line in f:
            a[int(line.split()[0]) - 1][int(line.split()[1]) - 1] = int(line.split()[2])
    df = pd.DataFrame(a)
    df.to_csv(outputf)

    return a


def build_df(inputf, outputf):
    df_file = open(outputf, mode="w")
    with open(inputf) as f:
        for line in f:
            row_dt = []
            row_dt = [item for item in line.split()]
            df_file.write(",".join(row_dt))
            df_file.write('\n')
    df_file.close()
    df = pd.read_csv(outputf, sep=",", names=["user_id", "item_id", "rating"])
    i_u_df = df[["item_id", "user_id", "rating"]].sort_values(by=["item_id", "user_id"]).reset_index(drop=True)
    return i_u_df

    return a


raw_matrix = build_matrix("train.txt","raw_matrix.csv")
#print raw_matrix
sparse_raw_matrix = sparse.csr_matrix(raw_matrix)
# print raw_matrix[0]
item_user_df = build_df("train.txt", "user_item.csv")
cold_start_index = np.where(~raw_matrix.any(axis=0))[0]

rated_value_sum = np.sum(raw_matrix, dtype=np.int32, axis=1)

r = raw_matrix
#r[r > 0] = 1
rated_num_sum = np.sum(r, dtype=np.int32, axis=1)
# print rated_num_sum
# print item_user_df.duplicated().sum()

split_value = int(len(item_user_df) * 0.80)
train_data = item_user_df[:split_value]
test_data = item_user_df[split_value:]
'''
plt.figure(figsize = (12, 8))
ax = seaborn.countplot(x="rating", data=train_data)
ax.set_yticklabels([num for num in ax.get_yticks()])
plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
plt.show()
'''
no_rated_movies_per_user = item_user_df.groupby(by="user_id")["rating"].count().sort_values(ascending=False)
#print no_rated_movies_per_user.head()

no_ratings_per_movie = train_data.groupby(by="item_id")["rating"].count().sort_values(ascending=False)
#print no_ratings_per_movie.head()


def get_u_i_sparse_matrix(df):
    sparse_data = sparse.csr_matrix((df.rating, (df.user_id, df.item_id)))
    return sparse_data


train_u_i_data = get_u_i_sparse_matrix(train_data)
test_u_i_data = get_u_i_sparse_matrix(test_data)

average_rating = train_u_i_data.sum()/train_u_i_data.count_nonzero()
#print ("global every rating: {}".format(average_rating))


def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1
    #print sum_of_ratings
    no_of_ratings = (sparse_matrix != 0).sum(axis = ax).A1
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings


average_r_u = get_average_rating(train_u_i_data,True)
average_r_i = get_average_rating(train_u_i_data,False)



def compute_user_similarity(sparse_matrix, limit=944):
    row_index, col_index = sparse_matrix.nonzero()
    rows = np.unique(row_index)
    similar_arr = np.zeros(944*limit).reshape(944, limit)

    for row in rows[:limit]:
        sim = cosine_similarity(sparse_matrix.getrow(row), train_u_i_data).ravel()
        ##similar_indices = sim.argsort()[-limit:]
        ##similar = sim[similar_indices]
        similar_arr[row] = sim

    return similar_arr,rows


similar_user_matrix,row_ind = compute_user_similarity(train_u_i_data, 944)
'''
t_s_getr = train_u_i_data.getrow(1)
silmilar = cosine_similarity(t_s_getr,train_u_i_data).ravel()
sim_arr = np.zeros(944*944).reshape(944,944)
sim_arr[1] = silmilar
'''
reader = Reader(line_format='user item rating',rating_scale=(1, 5))
#data_t1 = Dataset.load_from_df(df_rd[["user", "item", "rating"]], reader)
u_i_data = Dataset.load_from_file("train.txt",reader)
trainset,testset = train_test_split(u_i_data,test_size=0.2)

test_ml = Dataset.load_builtin()
sim_options = {
    "name": ["msd", "cosine","pearson"],
    "min_support": [1,3,5],
    "user_based": [False, True],
}
sim_options1 = {
    "name":"msd",
    "min_support":1,
    "user_based":False,
}
test_k= 53

test_mk = 6
gs = GridSearchCV(KNNWithMeans,param_grid=sim_options,measures=["rmse"],cv=5)
trainning_set = u_i_data.build_full_trainset()
#print('number of users: ',trainset.n_users)
#print('number of items: ',trainset.n_items)
train_iids = list(trainning_set.all_items())
iid_convert = lambda x:trainning_set.to_raw_iid(x)
trainset_raw_iids = list(map(iid_convert,train_iids))

#gs.fit(u_i_data)
#print(gs.best_score["rmse"])
#print(gs.best_params["rmse"])
algo = KNNWithMeans(k = test_k,min_k=test_mk,sim_options=sim_options1)

algo.fit(trainning_set)


'''
for i in range(0,11):
    algo = KNNWithMeans(k=test_k_list[i],min_k=test_mk,sim_options=sim_options1)
    algo.fit(trainset)
    predictions = algo.test(testset)
    print accuracy.rmse(predictions)

gs.fit(trainset)
predictions = gs.test(testset)
accuracy.rmse(predictions)

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
gs.fit(u_i_data)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
'''
df_uir = pd.read_csv("train.txt",sep = ' ',names = ['user_id','item_id','rating'])
df_ui = df_uir[["user_id","item_id"]]
dic = df_ui.to_dict("split")
k_data = dic["data"]

df_make = pd.DataFrame()
ui_matrix = pd.read_csv("raw_matrix.csv",index_col = 0)
ui_matrix = ui_matrix.rename(index = lambda x: int(x)+1,columns=lambda x: int(x)+1)
ui_matrix

#df_uir.append({'user_id' :i, 'item_id' :j,'rating':algo.predict(uid=str(i),iid=str(j))},ignore_index=True)
new_df = pd.DataFrame(ui_matrix.stack())
nindex = new_df.reset_index().rename(columns={"level_0":"userid","level_1":"itemid",0:"rating"})
nindex = nindex.astype(str)

final_data = Dataset.load_from_df(nindex[['userid','itemid','rating']],reader)
NA, final_test = train_test_split(final_data,test_size=1.0)

predictions = algo.test(final_test)
est = [i.est for i in predictions]
ro_est = np.around(np.array(est)).tolist()
nindex['rating'] = ro_est
nindex = nindex.astype('int64')
np.savetxt('output.txt',nindex.values,fmt="%d")
