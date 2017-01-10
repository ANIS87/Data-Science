from __future__ import print_function
import os
import pandas
import matplotlib as mpl
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
import sys
import StringIO, csv
import matplotlib.dates as md
import dateutil
from collections import Counter
from sklearn.preprocessing import normalize
import collections
from operator import itemgetter
import Markov_Functions as Markov
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
print('\n ')

################################################

##    data loading and preparation   ##

################################################
print('Data loading ...100%\n ')
file_data="data_brut_Final.csv" 
df = Markov.read_data(file_data)
#df=df.dropna()

#date transformation 
YEAR=[]
MONTH=[]
DAY=[]
HOUR=[]
MINUTE=[]
SECOND=[]


#choose a user

liste_ID= list(set(df['track_id']))
id_user=random.choice(liste_ID)
print(id_user)
#id_user='track55462a3d7c9615.03036560'
id_user='track553f74f73099c5.49609494'
#id_user='track555b0e3a4e6c16.59989716'
#id_user='track54f8b91f113a74.05247273'
id_user='track555626c2286035.07030620'


df=df[(df['track_id'] ==id_user)]

##transform_date
cerated_at_TAB=df['created_at']
for xx in cerated_at_TAB:   
    Y,M,D,H,MI,S=Markov.transform_date(xx)
    YEAR.append(Y)   
    MONTH.append(M)
    DAY.append(D)
    HOUR.append(H)
    MINUTE.append(MI)
    SECOND.append(S)

df['year']=YEAR
df['month']=MONTH
df['day']=DAY
df['hour']=HOUR
df['minute']=MINUTE
df['second']=SECOND

index_fin_url=0
T=[]
URL=df['referrer'].astype(str)
for url in URL:
    if isinstance(url, str):
      T.append(Markov.first_url(url,index_fin_url))
    else: 
      T.append(url)

df['page_open']=T


name_columns=['track_id','created_at','browser','page_open','src','referrer','name','event' ]
df=Markov.cleaning_data(df,name_columns)
df=df.dropna()

########################

##    Markov model    ##

########################
intial_state =['Start']
States_event=['click']
States_browser, squence_browser, browser_as_int=Markov.data_to_list(df,'browser')
States_Marque, squence_Marque, Marque_as_int=Markov.data_to_list(df,'name')
States_web, squence_web, web_as_int=Markov.data_to_list(df,'page_open')
States_product, squence_product, product_as_int=Markov.data_to_list(df,'src')
States_EVENT, squence_EVENT, EVENT_as_int=Markov.data_to_list(df,'event')
States_URL, squence_URL, URL_as_int=Markov.data_to_list(df,'referrer')
URL_Product=States_URL+States_product
States_URL_Product=[]
[States_URL_Product.append(item) for item in URL_Product if item not in States_URL_Product]
list_product=[]
[list_product.append(item) for item in States_product if item not in States_URL]


###Tranistion probabilities

print('Calculation of  transition probabilities ...100%\n ')

##Start to browser 
element_frequ=Counter(df['browser'])
if 'nan' in element_frequ.keys():
  del element_frequ['nan']
if '-' in element_frequ.keys():
  del element_frequ['-']
if 'undefined' in element_frequ.keys():
  del element_frequ['undefined']
frequence=np.array(element_frequ.values()).astype(float)
browser_name=element_frequ.keys()
probabilite_browser=frequence/np.sum(frequence, axis=0)
#browser toproduct 
Matrix_browser_URL=Markov.transition_brow_url(df,States_browser, States_URL)
#Url to product 
Url_to_product=Markov.transition_proba_URL_diff_SRC(df,States_URL, States_product)
#product to product 
product_to_product=Markov.calcul_proba_transition_1(product_as_int)
product_to_event=Markov.transition_product_to_clikc(df,States_product)
event_to_product=Markov.transition_click_product(squence_product,EVENT_as_int, States_product)

print('Filling the transition matrix and making noramlization ...100%\n')
####remplissage
Total_states=intial_state + States_browser + States_URL_Product +States_event
size_xy=len( Total_states)
Marray = np.zeros((size_xy,size_xy))
Matrix_intial_browser=frequence
Current_Matrix=Markov.filling_insertion (Marray, Matrix_intial_browser, Total_states,intial_state, States_browser)
Current_Matrix=Markov.filling_insertion (Marray, Matrix_browser_URL, Total_states,States_browser, States_URL)
Current_Matrix=Markov.filling_insertion (Marray, Url_to_product, Total_states,States_URL, States_URL_Product)
Current_Matrix=Markov.filling_insertion (Marray, product_to_product, Total_states,States_product, States_product)
Current_Matrix=Markov.filling_insertion (Marray, product_to_event, Total_states, States_product, States_event)
Current_Matrix=Markov.filling_insertion (Marray, event_to_product, Total_states, States_event, States_product)

####normalization 
Transition_Matrix=normalize(Current_Matrix, axis=1, norm='l1').astype(float)
Transition_Matrix=np.around(Transition_Matrix, decimals=2)

### kill 0 value to 1 if don't have any new state 
Transition_Matrix=Markov.function_kill_0(Transition_Matrix)

####matrix to data frame
Name_product=Markov.get_name_of_product(df,States_product)
Name_URL_Product=States_URL_Product
Name_URL_Product[len(States_URL_Product)- len(Name_product)::]=Name_product
Name_URL_Product[0]=Name_URL_Product[0].split("//")[-1].split("/")[0]
names=intial_state + States_browser + Name_URL_Product + States_event
df_transition = pd.DataFrame(Transition_Matrix, index=names, columns=names)








### Test HMM
print('\n')
print('\n')
print('Test HMM: behavior of the web user with respect to the products given by a web page branding') 

#### Extraction  of Transitionmatrix, obsrevations, Emission 
observations=np.array(EVENT_as_int).astype(int)
initial_probabilities=normalize(np.sum(Url_to_product[:,3::], axis=1), axis=1, norm='l1').astype(float)
observation_probabilities=Markov.Obsrevations_product_to_click(df,States_product, States_EVENT)
observation_probabilities=normalize(observation_probabilities, axis=1, norm='l1').astype(float)
transition_probabilities=product_to_product

num_states=len(States_product)



A=transition_probabilities
T={}
for i in range(num_states):
    ai={}
    for j in range(num_states):
        ai[j]=A[i,j]
    T[i]=ai


B= observation_probabilities
E={}  
for i in range(num_states):
    ai={}
    for j in range(2):
        ai[j]=B[i,j]
    E[i]=ai
print('\n')
Yi=initial_probabilities[0]
print(Yi)
I={}  
print(initial_probabilities)   
for i in range(num_states):
    I[i]=Yi[i]


    
print('Test my code \n:')

obs=[0,0,0,1]
#0: view
#1: click 

A,B=Markov.evaluate_beta(num_states, initial_probabilities, transition_probabilities, observation_probabilities, obs)
print(A)
print(B)

A,B=Markov.learning_HM(obs, num_states, initial_probabilities,observation_probabilities, transition_probabilities)
print(A)



