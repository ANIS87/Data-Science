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
import data_mining_libraries as training_librairie
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import my_url_object 

class matrix_transition(object):
   
  def __init__(self,dataframe_json_file,Features, Number_of_users):
     self.file_json=dataframe_json_file
     self.df= pd.read_csv(self.file_json)
     self.N=Number_of_users
     self.data_frame=self.df[0:self.N] 
     self.user_id=list(set(self.data_frame['track_id']))
     self.Features=Features
 
  
    
 
  def get_temporal_parameters(self):
      cerated_at_TAB=self.data_frame['created_at']
      YEAR=[],MONTH=[],DAY=[],HOUR=[],MINUTE=[],SECOND=[]
      for xx in cerated_at_TAB:   
        Y,M,D,H,MI,S=training_librairie.transform_date(xx)
        YEAR.append(Y)   
        MONTH.append(M)
        DAY.append(D)
        HOUR.append(H)
        MINUTE.append(MI)
        SECOND.append(S)

      self.data_frame['year']=YEAR
      self.data_frame['month']=MONTH
      self.data_frame['day']=DAY
      self.data_frame['hour']=HOUR
      self.data_frame['minute']=MINUTE
      self.data_frame['second']=SECOND

  def get_data_given_id(self,id_user):
      
      self.data=self.data_frame[(self.data_frame['track_id'] ==id_user)]
      self.data=training_librairie.cleaning_data(self.data,self.Features)
      self.data=self.data.dropna()
      return self.data 

  def get_elementary_behaviour(self,id_user):
     if id_user==None:
         id_user=random.choice(self.user_id)
     
     self.data=self.get_data_given_id(id_user)
     self.elementary_behaviour={}
     for feature in self.Features:
       dict_feature={}
       #this function is to represent the user behavior as a sequence of int or vocabularies : example, [ view, click, view]==>[0,1,0] wehre States are click or view 
       States, vocabulary_sequence, int_sequence=training_librairie.data_to_list(self.data, feature)  
       dict_feature['States']=States
       dict_feature['vocabulary_sequence']=vocabulary_sequence
       dict_feature['int_sequence']=int_sequence
       self.elementary_behaviour[feature]=dict_feature
     return  self.elementary_behaviour   

  def get_all_states(self,id_user):
     self.elementary_behaviour=self.get_elementary_behaviour(id_user)
     self.initial_states=['Start_session']
     self.browser_choosed=self.elementary_behaviour['browser']['States']
     self.web_session=self.elementary_behaviour['referrer']['States']
     self.visited_product=self.elementary_behaviour['src']['States']
     self.States_click=['click']
     self.States_view=['view']
     self.States_event=['view','click']
     self.URL_Product=self.web_session+self.visited_product
     self.States_web_sessison_and_visited_product=[]
     [self.States_web_sessison_and_visited_product.append(item) for item in self.URL_Product if item not in self.States_web_sessison_and_visited_product]
     
     self.all_states=self.initial_states + self.browser_choosed + self.States_web_sessison_and_visited_product +self.States_event
     self.size_xy=len( self.all_states)
     self.Transition_Matrix = np.zeros((self.size_xy,self.size_xy))
     
     return 
      
  def calculate_transition_probabilities(self, id_user):
      self.get_all_states(id_user)
      #browser:
      browser_frequencie=training_librairie.clean_nan_values(self.data, 'browser')
      self.frequence=np.array(browser_frequencie.values()).astype(float)
      browser_name=browser_frequencie.keys()
      self.Matrix_start_browser=self.frequence/np.sum(self.frequence, axis=0)  

      #browser to web session 
      self.Matrix_browser_URL=training_librairie.transition_brow_url( self.data,  self.browser_choosed,  self.web_session)

      #web session to product
      self.Matrix_Url_to_product=training_librairie.transition_proba_URL_diff_SRC( self.data,  self.web_session,      self.visited_product)        
      self.Matrix_Url_to_product=normalize(self.Matrix_Url_to_product, axis=1, norm='l1').astype(float)
      #product to product 
      self.Matrix_product_to_product=training_librairie.calcul_proba_transition_1( self.elementary_behaviour['src']['int_sequence']) 
      self.Matrix_product_to_product=self.function_kill_0(self.Matrix_product_to_product)
      self.Matrix_product_to_product=normalize(self.Matrix_product_to_product, axis=1, norm='l1').astype(float)
      #product to event
      self.Matrix_product_to_event_click=training_librairie.transition_product_to_click( self.data,self.visited_product)      
      self.Matrix_product_to_event_view=training_librairie.transition_product_to_view( self.data,self.visited_product)
      self.Matrix_product_to_event=np.concatenate(( self.Matrix_product_to_event_view,self.Matrix_product_to_event_click),axis=1)
      self.Matrix_product_to_event=self.Matrix_product_to_event.reshape((len(self.visited_product), 2))
      self.Matrix_product_to_event=normalize(self.Matrix_product_to_event, axis=1, norm='l1').astype(float)
      
      return   

  #add elemnt_matrix to the final matrix 
  def filling_insertion(self, elemnt_matrix,state_I, state_J):    
    ii=self.all_states.index(state_I[0])
    jj=self.all_states.index(state_J[0])  
    I0= ii
    I1=ii+len(state_I)
    J0= jj
    J1= jj+len(state_J)
    self.Transition_Matrix [I0:I1, J0:J1]=elemnt_matrix
    return  

  #if we don't have any transition from a such  state i to a state j we remain  at state i
  def function_kill_0(self,M):
     for i in range(M.shape[0]):
       a=M[i,:]
       if (a.sum()==0):
          M[i,i]=1
     return M 

  #Final matrix should be used then. 
  def get_transition_matrix(self,id_user ):
     self.calculate_transition_probabilities( id_user)
     self.filling_insertion(  self.Matrix_start_browser,       self.initial_states,    self.browser_choosed)
     self.filling_insertion(  self.Matrix_browser_URL,         self.browser_choosed,   self.web_session)
     self.filling_insertion(  self.Matrix_Url_to_product,      self.web_session,       self.States_web_sessison_and_visited_product )
     self.filling_insertion(  self.Matrix_product_to_product,  self.visited_product,   self.visited_product )
     self.filling_insertion(  self.Matrix_product_to_event_click,    self.visited_product,   self.States_click )
     self.filling_insertion(  self.Matrix_product_to_event_view,    self.visited_product,   self.States_view)
     self.Transition_Matrix=self.function_kill_0(self.Transition_Matrix)
     self.Transition_Matrix=normalize(self.Transition_Matrix, axis=1, norm='l1').astype(float)
     self.Transition_Matrix=np.around(self.Transition_Matrix, decimals=2)
     return self.Transition_Matrix
  #show matrix as dictionnary A[state_i][state_j]=aij     
  def visualize_matrix(self,id_user):
      self.Transition_Matrix=self.get_transition_matrix(id_user ) 
      self.Final_Matrix={}
      for state_i in self.all_states:
         U={}
         i=self.all_states.index(state_i)
         for state_j in self.all_states:
            j=self.all_states.index(state_j)
            U[state_j]=self.Transition_Matrix[i,j]
         self.Final_Matrix[state_i]=U

      return self.Final_Matrix,  self.Transition_Matrix     
     


  
