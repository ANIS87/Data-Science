from __future__ import print_function
import os
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.preprocessing import normalize
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
             "master/pandas/tests/data/data_shop_CPC.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download juste_pour_tester.csv")

        with open("data_brut.csv", 'w') as f:
            print("-- writing to local data_shop_tree file")
            df.to_csv(f)

    return df

##duplicates



#read data  form csv file
 
start_time = time.time()
df = get_iris_data()



for tt in range(1):
   print("features:")
   zo=['cpc','name']
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

print("result :")
#print(Ao[cpc=0.01])

tab_0__dot_0_1=Ao.loc[0.01]
tab_0__dot_0_2=Ao.loc[0.02]
tab_0__dot_0_3=Ao.loc[0.03]
tab_0__dot_10=Ao.loc[0.10]
tab_0__dot_25=Ao.loc[0.25]

##Verify   size 

#print(len(tab_0__dot_0_1)+len(tab_0__dot_0_2)+len(tab_0__dot_0_3)+len(tab_0__dot_10)+len(tab_0__dot_25)) cheek size


##0.01

A=tab_0__dot_25
name_pro=list(A.index)
#print(name_pro)

T=sorted(A)

S=sorted(range(len(A)), key=lambda k: A[k])

#ziw=A.sum()
ziw=1
T=np.array(T)
T=T.astype(float)
T=T[::-1]/ziw
T=T.astype(float)
S=S[::-1]
##sdfl

name_product=[]
for i in range(len(S)):
  name_product.append(name_pro[S[i]])
#print("size of each prodcut")
#print(T)
#print("name product")
#print(name_product)

name_product=np.array(name_product).astype(str)
name_product= [i.decode('utf-8') for i in name_product]

T=np.array(T).astype(float)
ee=min(len(A),15)
###figure
width = 0.35 
fig=plt.figure(10)
fig.clf()
plt.title("Statistics about CPC=0.25 for  each product ")
plt.bar(range(ee), T[0:ee], width = 0.20, align="center", color="b",label='0.25')

plt.legend()

plt.subplots_adjust(bottom=0.3)
#plt.bar(range(10), importances[indices][0:9],
       #color="r", yerr=std[indices][0:9], align="center")
#name_feat=[]
#VV=X_ind.tolist()
#for i in range(ee):
#  name_feat.append( input_feat[VV.index(indices[i])])

plt.xticks(rotation=90)
plt.xticks(range(ee), name_product[0:ee],fontsize=12)


plt.grid(10)

fig.savefig('Statistics_0_25.png') 

plt.show()








