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
import re, math 
WORD = re.compile(r'\w+')


##Markov functions###


####Data Mining 

def read_data(xx):    
    if os.path.exists(xx):        
        df = pd.read_csv(xx)    
    return df

def cleaning_data(df,name_columns):
   A= pd.DataFrame()
   for nc in name_columns:
       A[nc]=df[nc]
   return A 

def clean_nan_values(df, feature):
   element_frequ=Counter(df[feature])
   if 'nan' in element_frequ.keys():
       del element_frequ['nan']
   if '-' in element_frequ.keys():
      del element_frequ['-']
   if 'undefined' in element_frequ.keys():
      del element_frequ['undefined']
   return element_frequ

def transform_date(datestring):
  yourdate = dateutil.parser.parse(datestring)
  year_features=yourdate.year
  month_features=yourdate.month
  day_features=yourdate.day
  hour_features=yourdate.hour
  minute_features=yourdate.minute
  second_features=yourdate.second
  return year_features, month_features, day_features, hour_features,minute_features,second_features

#data segmentaion

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return days, hours, minutes, seconds

def date_slection (df, seuil):
  U=list(df['created_at'])
  j=0
  date_couple=[]
  index_couple=[]
  for M in U:
    start_date=M
    id_M=int(np.where(df['created_at']==M)[0])
    U.remove(M)
    end_date=0
    d1=dateutil.parser.parse( start_date)
    i=0
    j=j+1
    for N in U:
      d2=dateutil.parser.parse(N)
      
     
      T=abs((d2 - d1))
      d,h,m,s=convert_timedelta(T) 
     
      while (d==0 and h==0 and m<3 and s>30 and i<=len(U)):
        i=i+1
        end_date=N
        id_N=int(np.where(df['created_at']==N)[0])
    date_couple.insert(j,(start_date, end_date))
    index_couple.insert(j,(id_M, id_N))
  return date_couple, index_couple
       
##function URL 1

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
###
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

##function URL 2
def segmentation_URL(url,index_fin):
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


    return mm

def segmentation_URL(url):    
    mm=[]
    jj=[] 
    res=[]
    for j in range(0, len(url)):
        if (url[j] == '/' and url[j] != '//'):    
            mm.append(j)
    for j in range(0, len(url)):
        if (url[j] == '#' ):    
          jj.append(j)
    u=0
    if (len(mm)>0):
      for i in range(len(mm)-2):
        res.insert(u,url[mm[i+1]+1:mm[i+2]])
        u+=1
      
    else:
      res=url
    if len(jj)>0 :  
      res.insert(u, url[mm[len(mm)-1]+1:jj[0]]) 
      res.insert(u+1, url[jj[0]+1::]) 
      
    return res
######
def cheek_list(L):
    F=[]
    I=[]
    S=[]
    xxx='is float'
    if(len (L) >0):
        for i in L:      
           F.append(isinstance(i,float))
           I.append(isinstance(i,int))
           S.append(isinstance(i,str))
        ff,ii,ss=F.count(True),I.count(True),S.count(True)
        a=max(ff,ii,ss)
        if (ff==a):
           xxx= 'is float'
        elif (ii==a):
            xxx= 'is int'
        else:
             xxx= 'is str'
    return xxx
#############
#distance
##############
def distance_list_of_string(L,M):
    ll=len(L)
    mm=len(M)
    S=0.0
    if(ll*mm !=0):
       if ll >=mm:
         p=ll//mm
         Umax=L
         Umin=M
       else:
         p=mm//ll
         Umax=M
         Umin=L
       for i in range(len(Umin)):
           for j in range(p):
              if Umin[i]==Umax[i+j]:
                 S=S+1  
       d=float(((p+0)*len(Umin)-1*S)/((p+0)*len(Umin)))
    else:
       d=1.0
    return abs(round(d,2)) 

###
def distance_vector_of_int(user_1, user_2):
        sum1 = sum([x**2 for x in user_1])
        sum2 = sum([x**2 for x in user_2])
        numerator =0.0
        L=min(len(user_1), len(user_2))        
        for i in range(L):
            if (user_1[i]==user_2[i]):
               numerator=numerator+float(user_1[i]*user_2[i])       
        denominator = math.sqrt(sum1) * math.sqrt(sum2)        
        if (denominator==0.0 and numerator!=0.0):
            sim= 0.0
        elif (denominator==numerator):
            sim=1.0
        else:
            sim= float(round(numerator,2) / round(denominator,2))
        distance=float(math.acos(sim)/math.pi)
        k=1.0-sim
        return round(k,2)

#######

def distance_vector_of_float(user_1, user_2):
        sum1=0        
        Lmin=min(len(user_1), len(user_2)) 
        Lmax=max(len(user_1), len(user_2))
               
        for i in range(Lmin):
            sum1=sum1+float((user_1[i]-user_2[i])**2)
                      
        dis = math.sqrt(sum1)       
        if (Lmin==0 and Lmax !=0):
            d= 1
        elif (Lmin==0 and Lmax==0):
            d=0
        else:
            d= float(round(dis,2))       
        
        return round(d,2)

####### 
##Transition 
##########
def From_url_product_name(my_url):
   indices = [i for i, x in enumerate(my_url) if x == "/"]
   chaine=my_url[indices[-1]+1::]
   dot_index=[i for i, x in enumerate(chaine) if x == "."]
   p_index=[i for i, x in enumerate(chaine) if x == "#"]
   
   if (len(p_index) >0):
       product='product:'+chaine[p_index[0]::]
   elif (len(dot_index) >0):
      product=chaine[0:dot_index[0]]
   else:
      product=first_url(my_url,0)
  
   return product

###############
def data_to_dic(df1,name_columns,user_id):  
      
    df=df1[(df1['track_id'] ==user_id)]     
    df=cleaning_data(df,name_columns)
    df=df.dropna()
    para_time=list(df['created_at'])
    data_time=sorted(para_time)   
    RES={}
    for idx in name_columns:
         data_product=df[idx]       
         final_sequence=[]
         for tt in data_time:
            h=df[(df['created_at'] ==tt)]
            d= list(h[idx])      
            del h
            ii=data_time.index(tt)
            final_sequence.insert(ii,d[0])
         V=final_sequence
         R=[]
         [R.append(item) for item in V if item not in R] 
         int_list=[]
         ii=0
         for kk in V:
           ii=ii+1
           jj=R.index(kk)
           int_list.insert(ii,jj) 
         RES[idx]=V
    
    del df   
    return  RES  

###data to list without changing order !

def data_to_list(df,idx):
    data_product=df[idx]
    data_time=list(df['created_at'])
    data_time=sorted(data_time)
    final_sequence=[]
    for tt in data_time:
        h=df[(df['created_at'] ==tt)]
        d= list(h[idx])      
        del h
        ii=data_time.index(tt)
        final_sequence.insert(ii,d[0])
    V=final_sequence
    R=[]
    [R.append(item) for item in V if item not in R] 
    int_list=[]
    ii=0
    for kk in V:
     ii=ii+1
     jj=R.index(kk)
     int_list.insert(ii,jj)  
       
    return R,V, int_list

def get_name_of_product(df,States_product):
    H=[]
    t=0
    for jj in States_product:
        nameM=list(df[(df['src'] ==jj)]['name']) 
        H.insert(t,nameM[0])
        t=t+1
    return H

####function 1 and 2  calculate tranistion probabilities between to states at the same level 
def calcul_proba_transition_1(T):
  P=[]
  A=list(set(T)) 
  for i in A:
    Pij=[]

    for j in A:      
      item_i= find_index(T,i)
      aij=0      
      for h in item_i:
         if ( h< len(T)-1 and T[h+1]==j):
            aij=aij+1
      Pij.append(aij)
     
    P.append(Pij)

  U=np.array(P).astype(float)
  U=U.reshape(len(A),len(A))  
  Matrix_not_noramlized=U
  return Matrix_not_noramlized 
  return P

def find_index(X,k):
  H=[]
  for l in range(len(X)):
   if (X[l]==k):
     H.append(l)
  return H
###browser to URL
def transition_brow_url(df,States_browser, States_URL):    

  U=[]
  for hh in States_browser: 
    df_b=df[(df['browser'] ==hh)]
    for k in States_URL: 
      A=list(df_b['referrer']) 
      U.append(A.count(k))
    del df_b
  U=np.array(U).astype(float)
  U=U.reshape(len(States_browser),len(States_URL))
  
  Matrix_not_noramlized=U
  return Matrix_not_noramlized

###url to product 
def transition_proba_URL_diff_SRC(df,States_URL, States_product):   
  list_URL=[x for x in States_URL if str(x) != 'nan']
  list_SRC=[x for x in States_product if str(x) != 'nan']
  K=list_URL+list_SRC
  list_URL_SRC=[]  
  [list_URL_SRC.append(item) for item in K if item not in list_URL_SRC]

  U=[]
  for hh in list_URL: 
    df_b=df[(df['referrer'] ==hh)]
    for k in list_URL_SRC: 
      A=list(df_b['src']) 
      U.append(A.count(k))
    del df_b
  U=np.array(U).astype(float)
  U=U.reshape(len(list_URL),len(list_URL_SRC))
  
  Matrix_not_noramlized=U
  return Matrix_not_noramlized

##event click
def transition_product_to_clikc(df,States_product):
    U=[]
    for hh in States_product:
        df_b=df[(df['src'] ==hh)]
        A=list(df_b['event']) 
        U.append(A.count('click'))
        del df_b
    U=np.array(U).astype(float)
    U=U.reshape(len(States_product),1)
    return U 
###event

def Obsrevations_product_to_click(df,States_product, States_EVENT):
    U=[]
    for hh in States_product:
        df_b=df[(df['src'] ==hh)]
        for tt in States_EVENT:
          A=list(df_b['event']) 
          U.append(A.count(tt))
        del df_b
    U=np.array(U).astype(float)
    U=U.reshape(len(States_product),len(States_EVENT))
    return U         

##event click
  
def transition_click_product(squence_product,EVENT_as_int, States_product):
    size_x=len(States_product)
    B= np.zeros((1,size_x))
    A=B
    for h in range(len(EVENT_as_int)-1):
        if (EVENT_as_int[h]==1  and  EVENT_as_int[h+1]==0 ):
           k=States_product.index(squence_product[h])
           A[k]=A[k]+1
    return B 


###remplissage matrix

def filling_insertion (Curent_Matrix, elemnt_matrix, Total_parameters,state_I, state_J):
    ii=Total_parameters.index(state_I[0])
    jj=Total_parameters.index(state_J[0])  
    I0= ii
    I1=ii+len(state_I)
    J0= jj
    J1= jj+len(state_J)
    Curent_Matrix[I0:I1, J0:J1]=elemnt_matrix
    return Curent_Matrix


def function_kill_0(Curent_Matrix):
   for i in range(Curent_Matrix.shape[0]):
     a=Curent_Matrix[i,:]
     if (a.sum()==0):
          Curent_Matrix[i,i]=1
   return Curent_Matrix


#### functions to evalute probability of each sequence clik_view 
##num_states: number of states : number of visited web page for each session 
##initial_probabilities: probabilities to connect to a web page (of product)
##transition_probabilities: probability to change the page i to the page j 
##observation_probabilities : p(click) and p(view) for each page : page visted 1, page visited 2, ...., page visited N 
### obsrevations: give as sequence view view view event ... 
##transition matrix if number states=3  [* * *]                                                                       [* *]
#                                       [* * *] and Obsrevations porbbailities(matrix emession with 2 obs true/false )[* *] 
#                                       [* * *]                                                                       [* *]
##alpha
def evaluate_alpha(num_states, initial_probabilities, transition_probabilities, observation_probabilities, observations):     
  observation_length = len(observations)
  alpha = np.zeros([observation_length, num_states])

  for i in range(0, num_states):
     alpha[0, i] = initial_probabilities[0, i] * observation_probabilities[i, observations[0]]

  for t in range(1, observation_length):
    for j in range(0, num_states):
        bj=observation_probabilities[j, observations[t]]
        for i in range(0, num_states):          
           aij=transition_probabilities[i, j]
           alpha_i_1=alpha[t - 1, i]
           alpha[t, j] = alpha[t, j]+alpha_i_1 * aij*bj
  pp=float(alpha[observation_length - 1,:].sum())
  return pp, alpha





###beta
def evaluate_beta(num_states, initial_probabilities, transition_probabilities, observation_probabilities, observations):     
  observation_length = len(observations)
  beta = np.zeros([observation_length, num_states])
  pp=0
  for i in xrange(0, num_states):
     beta[observation_length - 1, i] = 1

  for t in reversed(range(observation_length-1)):
    for j in xrange(0, num_states):        
        for i in xrange(0, num_states):
           bi=observation_probabilities[i, observations[t+1]]          
           aji=transition_probabilities[j, i]
           beta_i_1=beta[t + 1, i]
           beta[t, j] = beta[t, j]+beta_i_1 * aji*bi 
     
  for j in xrange(0, num_states):
     pp=pp+initial_probabilities[0, j]*beta[0, j]*observation_probabilities[j, observations[0]]

  
  return pp, beta 
###   
def learning_HM(state_sequence, observations, num_states, initial_probabilities,observation_probabilities, transition_probabilities):
    print(observations)
    observation_length = len(observations)
    num_obsrv = len(list(set(observations)))
    A=transition_probabilities
    A=np.random.rand(num_states,num_states).astype(float)
    B=observation_probabilities
    B=np.random.rand(num_states,num_obsrv).astype(float)
    pi=np.zeros([num_states])
    delta=np.zeros([observation_length , num_states])
    xeta=np.zeros([observation_length, num_states , num_states]) 
    A=normalize(A, axis=1, norm='l1').astype(float)       
    B=normalize(B, axis=1, norm='l1').astype(float)   
    zz=1
    for jjj in range(10):
       
       A=normalize(A, axis=1, norm='l1').astype(float)
       KK=A
       B=normalize(B, axis=1, norm='l1').astype(float)  
       pobs,alpha=evaluate_alpha(num_states, initial_probabilities, A, B, observations)       
       pbet, beta=evaluate_beta(num_states, initial_probabilities, A, B, observations)
       for t in xrange(0, observation_length-1):
           for i  in xrange(0, num_states) : 
               alpha_t_i=alpha[t,i]
               beta_t_i=beta[t,i]
               
               if (pobs>0):
                   delta[t,i]=float((alpha_t_i* beta_t_i)/pobs)
               else:
                   delta[t,i]=0
               
               delta[observation_length-1,i]=float(alpha[observation_length-1,i]*beta[observation_length-1,i]/pobs)
               
               for j in xrange(0, num_states):
                   if(pobs>0):
                        xeta[t,i,j]=float(alpha[t,i]*beta[t+1,i]*A[i,j]*B[j,observations[t+1]]/pobs)
                   else:
                        xeta[t,i,j]=0 
                  
                   xeta[observation_length-1,i,j]=float(alpha[observation_length-1,i]*beta[observation_length-1,i]*A[i,j]*B[j,observations[observation_length-1]]/pobs)
                   
       for i  in xrange(0, num_states) :
           Sd=np.sum(xeta[:,i,:])
           for j  in xrange(0, num_states):
              Sx=0
              
              Sh=0
              for h in range(observation_length - 1):
                 Sx=Sx+xeta[h,i,j]
                 Sh=Sh+delta[h,i]
              
              if(Sd>0):  
                  A[i,j]=Sx/Sd
              else:
                 A[i,j]=Sx 
              
       zz=float(KK[2,2]-A[2,2])
       print(KK[2,2],A[2,2])
       for i  in xrange(0, num_states):
          rr=delta[:,i].sum()
          for u in xrange(0, num_obsrv): 
             c=0
             m=0
             for tt in xrange(0,observation_length):
                
                if  observations[tt]==u:
                   c=c+ delta[tt,i] 
             if (c==0):
                 
                 m=0
             else:
                 
                 m=float(c/rr)
               
             B[i,u]=m
       
    A=normalize(A, axis=1, norm='l1').astype(float) 
     
    B=normalize(B, axis=1, norm='l1').astype(float)    
    return A,B  
#https://github.com/ananthpn/pyhmm/blob/master/myhmm.py#L110


####

def decode_sequence_event(num_states, initial_probabilities, transition_probabilities, observation_probabilities, observations): 
    observation_length = len(observations)
    delta = np.zeros([observation_length, num_states])
    phi = np.zeros([observation_length, num_states])
    for i in xrange(0, num_states):
            delta[0, i] = initial_probabilities[0,i] * observation_probabilities[i, observations[0]]
            phi[0, i] = 0

    for t in xrange(1, observation_length):
            for j in xrange(0, num_states):
               bj=observation_probabilities[j, observations[t]]
               U=[]
               for i in xrange(0, num_states):
                   vi=delta[t-1,i]
                   aij=transition_probabilities[i, j]                 
                   avb = vi*aij
                   U.append(avb)
                   K=np.array(U)
               
               phi[t, j] = K.argmax() 
               delta[t, j] = K.max()*bj 
               
    state_sequence = np.zeros(observation_length).astype(int)
    state_sequence[observation_length - 1] = np.argmax(delta[observation_length - 1])
    sequence_probability = delta[observation_length - 1, state_sequence[observation_length - 1]]
    print ('s')
    print(phi)
    for t in reversed(xrange(0, observation_length - 1)):
         state_sequence[t] = phi[t+1, state_sequence[t + 1]] 
    return  state_sequence, sequence_probability, delta, phi 
###

