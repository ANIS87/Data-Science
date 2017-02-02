from __future__ import print_function
import os
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import pandas
import random
import numpy
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from urlparse import urlparse
import matplotlib.pyplot as plt

##clean data 
from sklearn.base import TransformerMixin
##variance
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing 
from sklearn.preprocessing import normalize
import time
from kmodes import kprototypes
import matplotlib.pyplot as plt

####



def get_iris_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("data_brut.csv"):
        print("-- data found locally")
        df = pd.read_csv("data_brut.csv")
    else:
        print("-- trying to download from github")
        fn = "https://raw.githubusercontent.com/pydata/pandas/" + \
             "master/pandas/tests/data/data_brut.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download data_brut")

        with open("data_brut.csv", 'w') as f:
            print("-- writing to local data_shop_tree file")
            df.to_csv(f)

    return df

##duplicates


def unique_columns2(data):
    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u,uind = np.unique(dataf, return_inverse=True)
    u = u.view(data.dtype).reshape(-1,data.shape[0]).T
    return (u,uind)
##remove
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# adresse IP
def transform_ip_domain (A):
  Z=[]
  for i in A:
    parsed_uri = urlparse( A[i] )
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
    Z.append(domain)
    return Z


#graph
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


#





def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pandas.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)










#read data 
start_time = time.time()
df = get_iris_data()
###print("* df.head()", df.head(), sep="\n", end="\n\n")
###print("* df.tail()", df.tail(), sep="\n", end="\n\n")
###print("* iris types:", df["_id"].unique(), sep="\n")
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

#encoding data
#_id,event,track_id,id_session,browser,browser_version,city,country,ip,resolution,hostname,time_arrival,pages_seen,referrer,time_spent,media,created_at,updated_at,latitude,longitude 

#df, _, _ = one_hot_dataframe(df, ['event'], replace=True)
#print (df)
#model



print("* df.head()", df.head(), sep="\n", end="\n\n")



Id_cols =[col for col in df.columns if 'track_id' in col]
event_col=[col for col in df.columns if 'event' in col]
time_arr=[col for col in df.columns if 'time_arrival' in col]
pages_seen_cols=[col for col in df.columns if 'pages_seen' in col]

time_cols = [col for col in df.columns if 'time_spent' in col]

count_cols = [col for col in df.columns if 'country' in col]

brow_cols = [col for col in df.columns if 'browser' in col]

med_cols = [col for col in df.columns if 'media' in col]
med_lat= [col for col in df.columns if 'latitude' in col]
med_log= [col for col in df.columns if 'longitude' in col]
crt_tim= [col for col in df.columns if 'created_at' in col]

##choose of input fetaures 

input_feat=Id_cols+pages_seen_cols+crt_tim+time_arr+time_cols+med_log+med_lat+event_col+count_cols+brow_cols+med_cols
print ("features input :")
print (input_feat)
X=df[input_feat]
print ("X_input :")
print(X[:][0:1])
print ("Z :")
NF=np.shape(X)
print("size of data")
print(NF[0])
N=int(NF[0])
Z1=np.array(X[:][0:10**9])
print ("max page seen:")
print (Z1[:, 0].max())
Z=np.array(X[:][0:N])
print(Z[:,0])
#classification of users 
Z[:, 0]=xrange(len(Z[:, 0]))
Z[:, 0] = Z[:, 0].astype(float)
Z[:, 1] = Z[:, 1].astype(float)
Z[:, 2] = Z[:, 2].astype(float)
Z[:, 3] = Z[:, 3].astype(float)
Z[:, 4] = Z[:, 4].astype(float)
Z[:, 5] = Z[:, 5].astype(float)
Z[:, 6] = Z[:, 6].astype(float)

##chose of element to classify and features 
user_id=Z[:, 0]
X_train=Z[:, [5,6,8]]

##decalage
X_train[:, 0]=X_train[:, 0]-X_train[:, 0].min()
X_train[:, 1]=X_train[:, 1]-X_train[:, 1].min()

##noralization 
X_train[:, 0]=X_train[:, 0]/(X_train[:, 0].max())
X_train[:, 1]=X_train[:, 1]/(X_train[:, 1].max())




del df 
print (user_id)
print ("X_train")
print (X_train)
#all data
#kproto = kprototypes.KPrototypes(n_clusters=3, init='Cao', n_init=1000, max_iter=10000, verbose=2)
#clusters = kproto.fit_predict(X_train, categorical=[6,7,8,9,10])
##countrie

##use of a KPrototypes algorithm to classify data 
kproto = kprototypes.KPrototypes(n_clusters=5, init='Cao', n_init=100, max_iter=1000, verbose=2)
clusters = kproto.fit_predict(X_train, categorical=[2])

# Print cluster centroids and categorical data mapping of the trained model.
print("Print cluster centroids : ")
print(kproto.cluster_centroids_)
centroids_list=kproto.cluster_centroids_
print("Print  categorical data mapping of the trained model:")
print(kproto.enc_map_)
# Print training statistics
print("Print training statistics:")
print(kproto.cost_)
print(kproto.n_iter_)

for s, c in zip(user_id, clusters):
    print("user: {}, cluster:{}".format(s, c))

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


K=centroids_list[0]
#print(K[:, :])
center_colors = colors[:len(K)]
#print (center_colors)
H=[]
for i in xrange(len(clusters)):
 #H[i]=center_colors[]

 l=clusters[i]
 #print(center_colors[l])
 H.append(center_colors[l])
print('color')
#print(H)
fig=plt.figure(2)
fig.clf()
##plot clusters with various colors

#plt.scatter(X_train[:, 0], X_train[:, 1], color='r')
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=50, linewidths=2,
            color=H, zorder=10)
plt.scatter(K[:, 0], K[:, 1], marker='x', s=100, linewidths=2,
            color='k', zorder=10)

plt.title('K-Prototypes clustering of visitors (longitude,latitude) \n'
          'Centroids are marked with black cross')
plt.grid(2)
plt.xlabel('longitude')
plt.ylabel('latitude')
fig.savefig('classification_k_prototypes_Assembla.png') 
#plt.show()
