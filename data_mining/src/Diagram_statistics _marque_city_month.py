from __future__ import print_function
import os
import pandas
import random
import numpy
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks
from string import letters
import dateutil.parser
import colorsys
####
def transform_date(datestring):
  yourdate = dateutil.parser.parse(datestring)
  year_features=yourdate.year
  month_features=yourdate.month
  hour_features=yourdate.hour
  return year_features, month_features, hour_features

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

index_fin_url=0



def read_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("data_brut_Final_name.csv"):
        print("-- data found locally")
        df = pd.read_csv("data_brut_Final_name.csv")
    else:
        print("-- trying to download from github")
        fn = "sssss" + \
             "master/pandas/tests/data/data_shop_CPC.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download juste_pour_tester.csv")

        with open("ddata_brut_Final_name.csv", 'w') as f:
            print("-- writing to local data_shop_tree file")
            df.to_csv(f)

    return df
 

df = read_data()
marq='Calvin Klein'



print ("Training ..")

YEAR=[]
MONTH=[]
HOUR=[]
##transform_date
cerated_at_TAB=df['created_at']
for xx in cerated_at_TAB:   
    Y,M,H=transform_date(xx)
    YEAR.append(Y)   
    MONTH.append(M)
    HOUR.append(H)
    

df['year']=YEAR
df['month']=MONTH
month_exist = list(set(df['month']))
df['hour']=HOUR

print ("Grouping ...")

for tt in range(1):
   print("features:")
   zo=['name','month','hour']
   
   freq_occuo=df.groupby(zo).count() 
   
   Ao=freq_occuo[:]['_id']
   
   
   
   
   indMaxo = freq_occuo['_id'].argmax()
   
   
   #df[zz]=df[zz].replace(np.nan,indMax, regex=True)

#print("month of buy :")
#print(month_exist )
#Mois
#tab_Janvier=Ao.loc[1]
#tab_Fevrier=Ao.loc[2]
#tab_Mars=Ao.loc[3]
#tab_Avril=Ao.loc[4]
#tab_Mai=Ao.loc[5]
#tab_Juin=Ao.loc[6]
#tab_Juillet=Ao.loc[7]
#tab_Aout=Ao.loc[8]
#tab_Sptember=Ao.loc[9]
#tab_October=Ao.loc[10]
#tab_November=Ao.loc[11]
#tab_December=Ao.loc[12]
##entreprise 
#tab_Adidas=Ao.loc['Adidas']

####Adidas for exemple ###"

Tab_adidas=Ao.loc[marq]
Tab_adidas_janvier=Tab_adidas.loc[1]
print("\n Adidas  satistics ... \n")
print(Tab_adidas)



T=np.array(Tab_adidas).astype(float)
AAA=T
print("table of year:")
print(len(AAA),float(len(AAA)/12))
#BBB=AAA.reshape(12,24)
#CCC=BBB.sum(axis=1)
#print("cheek :")
#print(AAA)
#print('\n')
#print(BBB)
#print('\n')
#print(CCC)
#area=CCC

##create table
L=[]
for ii in range(12):
 L.append(Tab_adidas.loc[ii+1].sum())


area=np.array(L).astype(float)


fig=plt.figure(10)
fig.clf()

fig.suptitle(marq, fontsize=20)
plt.subplot(221)
N = 12
# _, surface = zip(*surf) 
t=['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 
              'August', 'September', 'October', 'November', 'December'] 
surf=np.array(t).astype(str)
#surface = zip(*surf)
surface =surf

HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
       #Recuperation des nom des surface
col = RGB_tuples

 #Creation du diagramme et plot  utopct='%1.1f%%'

plt.pie(area / area.sum(),  labels=surface, colors=col,autopct='%1.1f%%',  startangle=90 )

plt.axis('equal')



print("\n Adidas  satistics Janvier... \n")
print(Tab_adidas_janvier)

T=np.array(Tab_adidas_janvier).astype(float)
area=T
plt.subplot(222)

# _, surface = zip(*surf) 
t=range(24) 
surf=np.array(t).astype(str)
#surface = zip(*surf)
surface =surf
N = 24
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
       #Recuperation des nom des surface
col = RGB_tuples

 #Creation du diagramme et plot  utopct='%1.1f%%'

plt.pie(area / area.sum(),  labels=surface, colors=col,autopct='%1.1f%%',  startangle=90 )
 
plt.axis('equal')




####city


def read_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("data_brut_stat.csv"):
        print("-- data found locally")
        df = pd.read_csv("data_brut_stat.csv")
    else:
        print("-- trying to download from github")
        fn = "sssss" + \
             "master/pandas/tests/data/data_shop_CPC.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download juste_pour_tester.csv")

        with open("data_brut_stat.csv", 'w') as f:
            print("-- writing to local data_shop_tree file")
            df.to_csv(f)

    return df
 
df = read_data()

for tt in range(1):
   print("features:")
   zo=['name','city']
   
   freq_occuo=df.groupby(zo).count() 
   
   Ao=freq_occuo[:]['_id']
   
   
   
   
   indMaxo = freq_occuo['_id'].argmax()



Tab_adidas=Ao.loc[marq]


A=Tab_adidas

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
N = 5
T=np.array(T).astype(float)
area=np.zeros(N)
area[0:N-1]=T[0:N-1]
area[N-1]=T[N-1::].sum()


plt.subplot(223)

# _, surface = zip(*surf) 
ii=[]
ii=name_product[0:N]
ii[N-1]='Remaining town'
surf=np.array(ii).astype(str)
#surface = zip(*surf)
surface =surf

HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
       #Recuperation des nom des surface
col = RGB_tuples

 #Creation du diagramme et plot  utopct='%1.1f%%'

plt.pie(area / area.sum(),  labels=surface, colors=col,autopct='%1.1f%%',  startangle=90 )
 
plt.axis('equal')

####references





def read_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("data_brut_stat_refer.csv"):
        print("-- data found locally")
        df = pd.read_csv("data_brut_stat_refer.csv")
    else:
        print("-- trying to download from github")
        fn = "sssss" + \
             "master/pandas/tests/data/data_shop_CPC.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download juste_pour_tester.csv")

        with open("data_brut_stat_refer.csv", 'w') as f:
            print("-- writing to local data_shop_tree file")
            df.to_csv(f)

    return df
 
df = read_data()
T=[]  
URL=df['referrer'].astype(str)
for url in URL:
    if isinstance(url, str):
      T.append(first_url(url,index_fin_url))
    else: 
      T.append(url)

df['referrer']=T




for tt in range(1):
   print("features:")
   zo=['name','referrer']
   
   freq_occuo=df.groupby(zo).count() 
   
   Ao=freq_occuo[:]['_id'] 
    
   
   indMaxo = freq_occuo['_id'].argmax()

Tab_adidas=Ao.loc[marq]


A=Tab_adidas

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
#name_product= [i.decode('utf-8') for i in name_product]
N = 6
T=np.array(T).astype(float)
area=np.zeros(N)
area[0:N-1]=T[0:N-1]
area[N-1]=T[N-1::].sum()
print("to test:")
print(T/T.sum())
print('\n')
print(name_product)
print('\n')
print(area/area.sum())
plt.subplot(224)

# _, surface = zip(*surf) 
ii=[]
ii=name_product[0:N]
ii[N-1]='Remaining websites'
surf=np.array(ii).astype(str)
#surface = zip(*surf)
surface =surf

HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
       #Recuperation des nom des surface
col = RGB_tuples

 #Creation du diagramme et plot  utopct='%1.1f%%'

plt.pie(area / area.sum(),  labels=surface, colors=col,autopct='%1.1f%%',  startangle=90 )
 
plt.axis('equal')

fig.savefig('Statistics_time_region_product.png')
 

