import numpy as np
import math 

'''
        __O1
x1   __
x2     02
       __03
'''
'''
une rgression lineaire exige la continuite mathematique et la convexite quadratique
si on travaille avec des lables discretes il est important de modeliser une telle response avec une fonction mathematique. Parmis les
fonctions les plus utilises on peut citer softmax, 1/(1+exp(-z)) qui se cractrisent par une reponse assez discrete (escalier). 
'''
#neurone yes or ok, we don't have a linear relation !!!!
#https://www.youtube.com/watch?v=9KM9Td6RVgQ

class Neural_Network(object):
  def __init__(self,x,y):
     self.data=x/np.amax(x,axis=0)    
     self.label=y
     self.label=self.label/100
     #design of network : model  architecture  
     self.number_layer=5
     self.input_lyaer=self.data.shape[1]     
     self.heden_layer_1=5 # number of neurone 
     self.heden_layer_2=5 
     self.heden_layer_3=5        
     self.output_layer=self.label.shape[1]

     self.W={}
     self.W[1]=np.random.randn(self.input_lyaer,self.heden_layer_1)
     self.W[2]=np.random.randn(self.heden_layer_1,self.heden_layer_2)
     self.W[3]=np.random.randn(self.heden_layer_2,self.heden_layer_3)
     self.W[4]=np.random.randn(self.heden_layer_3,self.output_layer)
     #training parameter
     self.alpha=0.2
     self.max_iteration =10**9
     self.epsilon=10**(-5)
     self.z={}
     self.a={}
     self.a[1]=self.data

  #Neuron activation 1 or zero 
  def sigmoid(self,z):
      return 1/(1+np.exp(-z))
  #prime function of sigmoid 
  def sigmoid_prime(self,z):
      #equal alsor to f*(1-f), where f=simgmoid
      return np.exp(-z)/(1+np.exp(-z))**2

  def error(self,a,b):
      vec_error=0
      for i in range(a.shape[0]):
         if (a[i] !=0):
            vec_error=vec_error+(1-b[i]/a[i])**2 
         else: 
             vec_error=vec_error+(a[i]-b[i])**2
      u=float(vec_error)/a.shape[0]
      #sum(cste_norm
      return u




  def forward_algorithme_rule_chaine(self):
      #connections (like convolution)
      self.z2=np.dot(self.data,self.W[1])
      self.z[2]= self.z2
      self.a2=self.sigmoid(self.z2)
      self.a[2]= self.a2
      #not linear but non-linear relation to characterize neuro response 
      
      self.z3=np.dot(self.a2,self.W[2])
      self.z[3]= self.z3
      self.a3=self.sigmoid(self.z3) 
      self.a[3]= self.a3
      self.z4=np.dot(self.a3,self.W[3])
      self.z[4]= self.z4
      self.a4=self.sigmoid(self.z4)
      self.a[4]= self.a4
      # new input==> new connections 
      self.z5=np.dot(self.a4,self.W[4])
      self.z[5]= self.z5
      #neuron final response : activation 
      self.label_hat=self.sigmoid(self.z5) 
      
      
      return np.round(self.label_hat,2)



  def cost_function_rule_chaine(self):
      #you use rule chaine aT_(l-1)* delta_l, where delta_l=W(L+1)T*delta_l*sigmoide_prime_l, and aT_(0)=data.transpose
      self.label_hat=self.forward_algorithme_rule_chaine()
      self.DJW={}
      
      self.delta={}
      self.delta[self.number_layer]=np.multiply(-(self.label-self.label_hat),self.sigmoid_prime (self.z[self.number_layer]) )
      
      self.DJW[self.number_layer-1]=np.dot(self.a[self.number_layer-1].T, self.delta[self.number_layer])
      
      for layer in range(3):
         
         self.delta[self.number_layer-(layer+1)]=np.dot(self.delta[self.number_layer-(layer+0)],\
         self.W[self.number_layer-(layer+1)].T)*self.sigmoid_prime (self.z[self.number_layer-(layer+1)])
         self.DJW[self.number_layer-(layer+2)]=np.dot(self.a[self.number_layer-(layer+2)].T, self.delta[self.number_layer-(layer+1)])
       
      return self.DJW






  def Backpropagation_algorithme_many_layer(self):
      #estimate W parameters 
      iter_number=0
      fin_condition=False 
      while (iter_number<self.max_iteration  and fin_condition==False):
      
          self.DJW=self.cost_function_rule_chaine()
          #update W 
          self.W[1]=self.W[1]-self.alpha*self.DJW[1]
          self.W[2]=self.W[2]-self.alpha*self.DJW[2]
          self.W[3]=self.W[3]-self.alpha*self.DJW[3]
          self.W[4]=self.W[4]-self.alpha*self.DJW[4]
          #take decision                     
          error_conver=self.error(self.label,self.label_hat)
          fin_condition=error_conver<self.epsilon          
          iter_number=iter_number+1
          


      return  self.label_hat, float(iter_number)/self.max_iteration





#use

size_test=1000
a=np.array(np.random.randint(10, size=(size_test,2))).astype(float) 
b=np.array(np.random.randint(90, size=(size_test,1))).astype(float) 


my_network=Neural_Network(a,b)
label_output=my_network.forward_algorithme_rule_chaine()
DJW=my_network.cost_function_rule_chaine()
A,B=my_network.Backpropagation_algorithme_many_layer()
K= np.concatenate((A, b/100), axis=1)
print(B)
