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
            exit("-- Unable to download juste_pour_tester.csv")

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

def somme_tabe(A,B):
 C=0
 for i in range(len(A)):
   C=C+A[i]*B[i]
 return C



#read data  form csv file
 
start_time = time.time()
df = get_iris_data()

print(list(df))

freq_occuo=df.groupby('cpc').count() 
print(freq_occuo)

#### URL segmentation for example : url=http://www.x.com/a/b/c ==> www.x.com/a/b
index_fin_url=1
T=[]

 
###
features_names=['event'] 

for tt in range(1):
   print("features:")
   zo=['event','cpc']
   print(zo)
   freq_occuo=df.groupby(zo).count() 
   print("statistques:")
   Ao=freq_occuo[:]['_id']
   print("somme:")
   print(Ao)
   print(len(Ao))
   
   indMaxo = freq_occuo['_id'].argmax()
   print("indMax:")
   print(indMaxo)
   #df[zz]=df[zz].replace(np.nan,indMax, regex=True)

print("statistiques:")
stat_click=Ao['click']
stat_view=Ao['view']
print("click:")
print(stat_click)
print("view:")
print(stat_view)
print("list:")
index_click_view=np.array(list(stat_click.index))
click_table=np.array(stat_click)
view_table=np.array(stat_view)
print(view_table)
print("somme_argent (centimes):")
print("view:", somme_tabe(view_table,index_click_view))
print("click:", somme_tabe(click_table,index_click_view))
ee=5

#time.sleep(100)  
width = 0.35 
fig=plt.figure(5)
fig.clf()
plt.title("Statistics about visitors: cpc click / view")
plt.bar(range(ee), click_table, width = 0.65, align="center", color="r",label='click')
plt.bar(range(ee), view_table, width = 0.45, align="center", color="g",label='view')

plt.legend()

plt.subplots_adjust(bottom=0.3)
#plt.bar(range(10), importances[indices][0:9],
       #color="r", yerr=std[indices][0:9], align="center")
#name_feat=[]
#VV=X_ind.tolist()
#for i in range(ee):
#  name_feat.append( input_feat[VV.index(indices[i])])

plt.xticks(rotation=90)
plt.xticks(range(ee), index_click_view[0:ee],fontsize=12)


plt.grid(5)

plt.savefig('Statistics_cpc_money.png') 
#plt.show()















