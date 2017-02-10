from __future__ import print_function
import os
import pandas
import random
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks
from string import letters

##read data 

def read_data():
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
 

df = read_data()


###data calassification 
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
   


#table for each cpc 

tab_0__dot_01=Ao.loc[0.01]
tab_0__dot_02=Ao.loc[0.02]
tab_0__dot_03=Ao.loc[0.03]
tab_0__dot_10=Ao.loc[0.10]
tab_0__dot_25=Ao.loc[0.25]

##for example cpc=0.25

A=tab_0__dot_03
name_pro=list(A.index)
T=sorted(A)
S=sorted(range(len(A)), key=lambda k: A[k])
T=np.array(T)
T=T.astype(float)
T=T[::-1]
T=T.astype(float)
S=S[::-1]
name_product=[]
for i in range(len(S)):
  name_product.append(name_pro[S[i]])


name_product=np.array(name_product).astype(str)
name_product= [i.decode('utf-8') for i in name_product]

T=np.array(T).astype(float)
 
###figure
width = 0.35 
fig=plt.figure(10)
fig.clf()
ee=min(len(A),30)
plt.title("Statistics about CPC=0.25 for  each product ")
plt.bar(range(ee), T[0:ee], width = 0.20, align="center", color="b",label='0.25')
plt.legend()
plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=90)
plt.xticks(range(ee), name_product[0:ee],fontsize=10)
plt.grid(10)
fig.savefig('Statistics_0_25.png')
#plt.show()






