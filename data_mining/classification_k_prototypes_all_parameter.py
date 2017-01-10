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
from pylab import pcolor, show, colorbar, xticks, yticks
from string import letters
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
            exit("-- Unable to download data_brut.csv")

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




#graph of classification 
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



def first_url(url,index_fin):
    K=[]
    mm=[]
    sym="//" 
    if (url.find(sym)!=-1):
      ee=url.index(sym)      
      url=url[ee+2:]
      ee=0
      if (url.find(sym)!=-1):
         ee=url.index(sym)      
         url=url[ee+2:]
         ee=0
      
    else:
      ee=-2
      url=url[ee+2:]
      ee=0
      
    #print("edex")
    #print(ee)
    for j in range(0, len(url)):
        if (url[j] == '/'):
            mm.append(j)
   

    if (len(mm)>index_fin):
        
        zz=mm[index_fin]
        
        
        
        K=url[ee:zz]
        
    else:
        hh=max(ee,0)
        K=url[hh:]


    return K









#read data  form csv file
 
start_time = time.time()
df = get_iris_data()



#### URL segmentation for example : url=http://www.x.com/a/b/c ==> www.x.com/a/b
index_fin_url=1
T=[]

 
###
features_names=['browser','browser_version','city','referrer'] 

for tt in range(4):
   print("features:")
   zz=features_names[tt]
   print(zz)
   freq_occu=df.groupby(zz).count() 
   #print(freq_occu)
   indMax = freq_occu['_id'].argmax()
   print("indMax:")
   print(indMax)
   df[zz]=df[zz].replace(np.nan,indMax, regex=True)



URL=df['referrer'].astype(str)
for url in URL:
    if isinstance(url, str):
      T.append(first_url(url,index_fin_url))
    else: 
      T.append(url)

df['referrer']=T
Z=df['referrer']
print("URL")
print(URL[:][1:10])
print("T")
print(Z[:][80:]) 


#time.sleep(60)  

print(len(np.unique(URL)))
print(len(np.unique(df['referrer'])))


## size of data should be used in this script 

#data_size=10**5
print("taille de df:")
print(len(df))
#df=df[1:data_size]


###print data 


print("* df.head()", df.head(), sep="\n", end="\n\n")
#time.sleep(60)
print("* df.tail()", df.tail(), sep="\n", end="\n\n")
print("* iris types:", df["_id"].unique(), sep="\n")
print("df:df:df")
print(df['referrer'][80:]) 
#df['referrer']=df['referrer'].replace('nan',' www.google.com', regex=True)
print("ca:ca:ca")
print(df['referrer'][80:]) 


#time.sleep(60) 



#_id,event,track_id,id_session,browser,browser_version,city,country,ip,resolution,hostname,time_arrival,pages_seen,referrer,time_spent,media,created_at,updated_at,latitude,longitude 
df['browser_version'] = df['browser_version'].astype(str)

#Encoding categorical features: create dummy variables

#Example : 
#  x0        X1           x0=USA   x0=FR   X1=Firefox   X1=chrome
# 'USA'   'firefox' ==>    1         0          1           0
# 'FR'    'chrome' 
#,'referrer'        0         1          0           1
df=df.rename(columns={'browser_version': 'web_version'})

#time.sleep(60)   

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

time_spent = [col for col in df.columns if 'time_spent' in col]

count_cols = [col for col in df.columns if 'country' in col]

city_cols = [col for col in df.columns if 'city' in col]


brow_cols = [col for col in df.columns if 'browser' in col]

med_cols = [col for col in df.columns if 'media' in col]
med_lat= [col for col in df.columns if 'latitude' in col]
med_log= [col for col in df.columns if 'longitude' in col]
crt_tim= [col for col in df.columns if 'created_at' in col]
ref_col= [col for col in df.columns if 'referrer' in col]

##choose of input fetaures 

input_feat=Id_cols+pages_seen_cols+time_arr+crt_tim+time_spent+med_log+med_lat+event_col+city_cols+count_cols+brow_cols+med_cols+ref_col
print ("features input :")
print (input_feat)
X=df[input_feat]
print ("X_input :")
print(X[:][0:1])
print ("Z :")
NF=np.shape(X)
N=int(NF[0])
Z1=np.array(X[:][0:N])
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
###
event=Z[:, 7]
browser=Z[:, 10]
###
X_train=Z[:, [3,4,5,6,7,8,9,10,11]]

##decalage
X_train[:, 0]=X_train[:, 0]-X_train[:, 0].min()
X_train[:, 1]=X_train[:, 1]-X_train[:, 1].min()
X_train[:, 2]=X_train[:, 2]-X_train[:, 2].min()
X_train[:, 3]=X_train[:, 3]-X_train[:, 3].min()
##noralization 
X_train[:, 0]=X_train[:, 0]/(X_train[:, 0].max())
X_train[:, 1]=X_train[:, 1]/(X_train[:, 1].max())
X_train[:, 2]=X_train[:, 2]/(X_train[:, 2].max())
X_train[:, 3]=X_train[:, 3]/(X_train[:, 3].max())

print("pour tester:")
print(X_train[:, 1].max(), X_train[:, 1].min())
del df 
print (user_id)
print ("X_train")
print (X_train)
#all data
#kproto = kprototypes.KPrototypes(n_clusters=3, init='Cao', n_init=1000, max_iter=10000, verbose=2)
#clusters = kproto.fit_predict(X_train, categorical=[6,7,8,9,10])
##countrie

##use of a KPrototypes algorithm to classify data 
kproto = kprototypes.KPrototypes(n_clusters=3, gamma=0.5,  init='Cao', n_init=100, max_iter=1000, verbose=2)
clusters = kproto.fit_predict(X_train, categorical=[4,5,6,7,8])

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
ATABA=[]
## print cluster of each user 
for e,b, c in zip(event, browser, clusters):
   if(e=='click' and b=='Safari'):
    ATABA.append([e,b,c])
print("occurnce:")
print(100*len(ATABA)/N)
print("clustring:")
print(ATABA)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


K=centroids_list[0]
print(K[:, :])
center_colors = colors[:len(K)]
print (center_colors)

H=[]
for i in xrange(len(clusters)):
 #H[i]=center_colors[]

 l=clusters[i]
 #print(center_colors[l])
 H.append(center_colors[l])
#print('color')
#print(H)

##plot clusters with various colors
fig=plt.figure(3)
fig.clf()
#plt.scatter(X_train[:, 0], X_train[:, 1], color='r')
plt.scatter(X_train[:, 2], X_train[:, 3], marker='o', s=50, linewidths=2,
            color=H, zorder=10)
plt.scatter(K[:, 2], K[:, 3], marker='x', s=100, linewidths=2,
            color='k', zorder=10)

#plt.title('K-Prototypes clustering of mixed data\n'
          #'Centroids are marked with white cross')
plt.grid(3)

fig.savefig('classification_k_prototypes_manyfetaures_REFERRER.png') 



#plt.show()
