import numpy as np
import math 
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
     self.input_lyaer=2
     self.heden_layer=3
     self.number_layer=3
     self.output_layer=1
     self.W1=np.random.randn(self.input_lyaer,self.heden_layer)
     self.W2=np.random.randn(self.heden_layer,self.output_layer)
     #training parameter
     self.alpha=0.2
     self.max_iteration =100000
     self.epsilon=10**(-5)
   

  #Neuron activation 1 or zero 
  def sigmoid(self,z):
      return 1/(1+np.exp(-z))

  def sigmoid_prime(self,z):
      return np.exp(-z)/(1+np.exp(-z))**2

  def error(self,a,b):
      vec_error=[(1-b[i]/a[i])**2 for i in range(a.shape[0])]
      
      #sum(cste_norm
      return float(sum(vec_error)/a.shape[0])

  def forward_algorithme(self):
      #connections (like convolution)
      self.z2=np.dot(self.data,self.W1)
      #not linear but non-linear relation to characterize neuro response 
      self.a2=self.sigmoid(self.z2)
      # new input==> new connections 
      self.z3=np.dot(self.a2,self.W2)
      #neuron response : activation 
      self.label_hat=self.sigmoid(self.z3)
      
      return np.round(self.label_hat,2)







  def cost_function(self):
      
      self.label_hat=self.forward_algorithme()
      self.diff=self.error(self.label_hat, self.label)
      ''' error cost : 1/2 sum(label- label_hat)^2, first given as raw data, second is given
       as a function of the network'design (w,input)
      rule chaine to explain the gradient with respect to both w parameters '''
      #you use rule chaine aT_(l-1)* delta_l, where delta_l=W(L+1)T*delta_l*sigmoide_prime_l, and aT_(0)=data.transpose
      delta3=np.multiply(-(self.label-self.label_hat),self.sigmoid_prime (self.z3) )
      DJW2=np.dot(self.a2.T, delta3)

      delta2=np.dot(delta3, self.W2.T)* self.sigmoid_prime (self.z2)
      DJW1=np.dot(self.data.T, delta2)
      return DJW1, DJW2

  def set_wights_update(self, wights):
      W1_start=0
      W1_end=self.heden_layer*self.heden_layer
      self.W1=np.reshape(wights[W1_start:W1_end], (self.input_lyaer,self.heden_layer))

      W2_end=W1_end+self.heden_layer*self.ouput_layer
      self.WZ=np.reshape(wights[W1_end:W2_end], (self.heden_layer,self.output_lyaer))

  
      


  def Backpropagation_algorithme(self):
      #estimate W parameters 
      iter_number=0
      fin_condition=False 
      while (iter_number<self.max_iteration  and fin_condition==False):
      
          dJDW1, dJDW2=self.cost_function()
          #update W 
          self.W1=self.W1-self.alpha*dJDW1
          self.W2=self.W2-self.alpha*dJDW2
          #take decision                     
          error_conver=self.error(self.label,self.label_hat)
          fin_condition=error_conver<self.epsilon          
          iter_number=iter_number+1
          


      return  self.label_hat,iter_number



#use

x=np.array([[3,5],[5,1],[10,2],[7,2]]).astype(float) 
y=np.array([[70],[25],[20],[30]]).astype(float) 
my_network=Neural_Network(x,y)
error=my_network.forward_algorithme()
a,b=my_network.cost_function()
prdicted_y,iter_number=my_network.Backpropagation_algorithme()
print(my_network.label)
print('\n')
print(my_network.label_hat)
print('\n')
print(iter_number)






