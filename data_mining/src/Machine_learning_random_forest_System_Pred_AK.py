from __future__ import print_function
import os
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import pandas
import random
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
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
import time

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


##URL Trasformation 
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
index_fin_url=0
T=[]

 
###
features_names=['browser','browser_version','city','country','referrer'] 

for tt in range(5):
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
data_size=(1679380-80)
print("taille de df:")
print(len(df))
df=df[1:data_size]


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
df, _, _ = one_hot_dataframe(df, ['event', 'referrer', 'browser','media'], replace=True)






###selection of input features  



time_arrival_parameter=[col for col in df.columns if 'time_arrival' in col]
pages_seen_parameter=[col for col in df.columns if 'pages_seen' in col]
time_spent_parameter= [col for col in df.columns if 'time_spent' in col]

latitude_parameter= [col for col in df.columns if 'latitude' in col]
longitude_parameter= [col for col in df.columns if 'longitude' in col]
country_parameter = [col for col in df.columns if 'country' in col]
city_parameter = [col for col in df.columns if 'city' in col]

browser_parameter = [col for col in df.columns if 'browser' in col]
media_parameter = [col for col in df.columns if 'media' in col]
referrer_parameter= [col for col in df.columns if 'referrer' in col]

cpc_parameter=[col for col in df.columns if 'cpc' in col]


### X 
input_feat=time_arrival_parameter+time_spent_parameter+latitude_parameter+longitude_parameter+browser_parameter+referrer_parameter+pages_seen_parameter+cpc_parameter
print ("features input :")
print (input_feat)

X=df[input_feat]

print ("X_input :")
print(X[:][0:1])

#### Y

Y=df['event=click']

del df

### data cleaning : replace nan values 
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
 
print ("no nan:")    
print ("....")
print ("....")  
X=imp.transform(X)
print(np.shape(X))

##Feature selection with respect to the  variance : remove features with lower variance 
sel = VarianceThreshold()
X=sel.fit_transform(X)
print ("no parametre lower variance:")    
print ("....")
print ("....") 

###Feature normalization with respect to the maximum
min_max_scaler = preprocessing.MinMaxScaler()
XK= min_max_scaler.fit_transform(X)
print("normalized data  ::::::: ")
print (XK[:][0:10])
print ("max hehe :")
print (np.max(XK[:,[0]]))
print (np.max(XK[:,[1]]))
print (np.max(XK[:,[2]]))
print (np.max(XK[:,[3]]))

X[:,[0,1,2,3]]=XK[:,[0,1,2,3]]
print("X normalized ::::::: ")
print (X[:][0:10])
#machine learning (30%)
##delete duplicates 
X[:,[0,1,2,3]]=XK[:,[0,1,2,3]]
print ("before_delete:", np.shape(X))
X,X_ind=unique_columns2(X)
print ("after_delete:", np.shape(X))
#print (X_ind)

a1=[]
a2=[]
b=[]

NF=np.shape(X)
data_size=NF[0]
print("data_size")
print(data_size)
pas_iter=int(data_size/100)
for i in range(80):
 
 t=i*pas_iter+1
 r=100*t/data_size
 b.append(r)
 X_train=X[:][0:t]
 Y_train=Y[:][0:t]
 rf = RandomForestClassifier(max_depth = 4)
 rf.fit(X_train, Y_train)
 X_test=X[:][t+1:]
 Y_test=Y[:][t+1:]
 #Z=rf.predict_proba(X_test)
 dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
 dt.fit(X_train, Y_train)
 Z1=dt.predict(X_test)
 Z2=rf.predict(X_test)
 y_true=Y_test
 y_pred_1=Z1
 y_pred_2=Z2
#print(np.shape(y_true))
 u1=accuracy_score(y_true, y_pred_1,normalize=True)
 u2=accuracy_score(y_true, y_pred_2,normalize=True)
 a1.append(u1)
 a2.append(u2)
 
print("Machine learning finiched ...")



#plot result 
fig=plt.figure(7)
fig.clf()
p1=plt.plot(b,a1, linewidth=2.0, color='r',label='DecisionTreeClassifier')
p2=plt.plot(b,a2,  linewidth=2.0,color='g',label='RandomForestClassifier')
plt.title('Decision tree classifier (shop_match): model quality')
plt.xlabel('n_samples (%) ')
plt.ylabel('accuracy_score')
#plt.text(1,35,'inputs:time_arrival,pages_seen, time_spent,media,created_at,updated_at,latitude,longitude ')
#plt.legend([p1, p2], ["DecisionTreeClassifier","DecisionTreeClassifier_2"],loc=1)
plt.grid(7)


print("--- %s minutes ---" % ((time.time() - start_time)/60))
plt.legend()
fig.savefig('Machine_learning.png') 
#plt.show()   








