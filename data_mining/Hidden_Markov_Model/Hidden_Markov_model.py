import numpy as np
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
#This code well be used to analyze the user's behaviour using the tracking data 


class Hidden_Markov_Model():
      def __init__(self, Transition_matrix, Emession_matrix, Initial_probabilities):
           self.initial_probabilities=np.array(Initial_probabilities).astype(float)
           self.transition_probabilities=np.array(Transition_matrix).astype(float)
           self.observation_probabilities=np.array(Emession_matrix).astype(float)
           self.num_states=self.transition_probabilities.shape[0]
           self.num_obsrv=self.observation_probabilities.shape[1]
           
           

      def round_matrix(self,X):
         m,n=X.shape
         for i in range(m):
           for j in range(n):
             X[i,j]=round(X[i,j],2)
         return X
      def evaluate_alpha(self, observations):     
        observation_length = len(observations)
        alpha = np.zeros([observation_length, self.num_states])

        for i in range(0, self.num_states):
           alpha[0, i] = self.initial_probabilities[i] * self.observation_probabilities[i, observations[0]]

        for t in range(1, observation_length):
          for j in range(0, self.num_states):
              bj=self.observation_probabilities[j, observations[t]]
              for i in range(0, self.num_states):          
                 aij=self.transition_probabilities[i, j]
                 alpha_i_1=alpha[t - 1, i]
                 alpha[t, j] = alpha[t, j]+alpha_i_1 * aij*bj
        pobs=float(alpha[observation_length - 1,:].sum())
        return round(pobs,2),alpha





###beta
      def evaluate_beta(self,observations):     
        observation_length = len(observations)
        beta = np.zeros([observation_length, self.num_states])
        pp=0
        for i in xrange(0, self.num_states):
           beta[observation_length - 1, i] = 1

        for t in reversed(range(observation_length-1)):
          for j in xrange(0, self.num_states):        
              for i in xrange(0, self.num_states):
                 bi=self.observation_probabilities[i, observations[t+1]]          
                 aji=self.transition_probabilities[j, i]
                 beta_i_1=beta[t + 1, i]
                 beta[t, j] = beta[t, j]+beta_i_1 * aji*bi 
     
        for j in xrange(0, self.num_states):
           pp=pp+self.initial_probabilities[j]*beta[0, j]*self.observation_probabilities[j, observations[0]]

  
        return  round(pp,2),beta 
###   
      def learning_HM(self, observations,Iter_max):
          
          observation_length = len(observations)
          
          A=self.transition_probabilities
          #A=np.random.rand(self.num_states,self.num_states).astype(float)
          B=self.observation_probabilities
          #B=np.random.rand(self.num_states,self.num_obsrv).astype(float)
          pi=np.zeros([self.num_states]).astype(float) 
          delta=np.zeros([observation_length , self.num_states])
          xeta=np.zeros([observation_length, self.num_states , self.num_states]) 
   
          iter_alg=0
          sum_pi=0
          change_matrix_test=False
          ant_zero=1
          while(iter_alg<Iter_max and change_matrix_test==False and ant_zero >0) :
             iter_alg=iter_alg+1           
             
             
             self.transition_probabilities=self.round_matrix(normalize(A, axis=1, norm='l1').astype(float))
             A_past=self.round_matrix(normalize(A, axis=1, norm='l1').astype(float))            
             B=normalize(B, axis=1, norm='l1').astype(float)
             self.observation_probabilities=self.round_matrix(normalize(B, axis=1, norm='l1').astype(float))   
             B_past=self.round_matrix(normalize(B, axis=1, norm='l1').astype(float))             
              
             p_alpha,alpha=self.evaluate_alpha(observations) 
             pobs=max (p_alpha,0.0001) #to avoid a "Division by zero" error 
                
             pbet, beta=self.evaluate_beta(observations)
             
            
             
             self.initial_probabilities=pi
             for t in xrange(0, observation_length-1):
                 for i  in xrange(0, self.num_states) : 
                     alpha_t_i=alpha[t,i]
                     beta_t_i=beta[t,i]
               
                     if (pobs>0):
                         delta[t,i]=round(float((alpha_t_i* beta_t_i)/pobs),2)
                     else:
                         delta[t,i]=0.0
                      
                     delta[observation_length-1,i]=round(float(alpha[observation_length-1,i]*beta[observation_length-1,i]/pobs),2)
               
                     for j in xrange(0, self.num_states):
                         pi[j]=delta[0,j]
                         if(pobs>0):
                              xeta[t,i,j]=round(float(alpha[t,i]*beta[t+1,i]*A[i,j]*B[j,observations[t+1]]/pobs),2)
                         else:
                              xeta[t,i,j]=0.0 
                  
                         xeta[observation_length-1,i,j]=round(float(alpha[observation_length-1,i]*beta[observation_length-1,i]*A[i,j]*B[j,observations[observation_length-1]]/pobs),2)
                   
             for i  in xrange(0, self.num_states) :
                 Sd=np.sum(xeta[:,i,:])
                 for j  in xrange(0, self.num_states):
                    Sx=0
              
                    Sh=0
                    for h in range(observation_length - 1):
                       Sx=Sx+xeta[h,i,j]
                       Sh=Sh+delta[h,i]
              
                    if(Sd>0):  
                        A[i,j]=Sx/Sd
                    else:
                       A[i,j]=Sx 
              
             
             
             for i  in xrange(0, self.num_states):
                rr=delta[:,i].sum()
                for u in xrange(0, self.num_obsrv): 
                   c=0
                   m=0
                   for tt in xrange(0,observation_length):
                
                      if  observations[tt]==u:
                         c=c+ delta[tt,i] 
                   if (c*rr==0):
                 
                       m=0.0
                   else:
                 
                       m=round(float(c/rr),2)
               
                   B[i,u]=m
             A_new=self.round_matrix(normalize(A, axis=1, norm='l1').astype(float))
             B_new=self.round_matrix(normalize(B, axis=1, norm='l1').astype(float))
             change_matrix_test=np.array_equal(A_new,A_past)
             ant_zero=float(A_new.sum()*B_new.sum()*pi.sum()) #to avoid divergence of iteration 
             

       
          A=normalize(A, axis=1, norm='l1').astype(float) 
          
          
          B=normalize(B, axis=1, norm='l1').astype(float)
          pi=normalize(pi, axis=1, norm='l1').astype(float)
              
          return A_past,B_past,iter_alg  


      def decode_sequence_event(self,observations): 
          observation_length = len(observations)
          delta = np.zeros([observation_length, self.num_states])
          phi = np.zeros([observation_length, self.num_states])
          for i in xrange(0, self.num_states):
                  delta[0, i] = self.initial_probabilities[i] * self.observation_probabilities[i, observations[0]]
                  phi[0, i] = 0

          for t in xrange(1, observation_length):
                  for j in xrange(0, self.num_states):
                     bj=self.observation_probabilities[j, observations[t]]
                     U=[]
                     for i in xrange(0, self.num_states):
                         vi=delta[t-1,i]
                         aij=self.transition_probabilities[i, j]                 
                         avb = vi*aij
                         U.append(avb)
                         K=np.array(U)
               
                     phi[t, j] = K.argmax() 
                     delta[t, j] = K.max()*bj 
               
          state_sequence = np.zeros(observation_length).astype(int)
          state_sequence[observation_length - 1] = np.argmax(delta[observation_length - 1])
          sequence_probability = delta[observation_length - 1, state_sequence[observation_length - 1]]
          
          
          for t in reversed(xrange(0, observation_length - 1)):
               state_sequence[t] = phi[t+1, state_sequence[t + 1]] 
          return  state_sequence
###


###use of code  

T=[[0.5,0.3,0.2],[0.4,0.3,0.3],[0.1,0.6,0.3]]
I=[0.2,0.6,0.2]
E=[[0.1,0.4,0.5],[0.5,0.5,0.0],[0.2,0.8,0.]]
T=np.array(T).astype(float)

A=Hidden_Markov_Model(T,E,I)
obs=[0,1,1,0,1,2,2]
seq=A.decode_sequence_event(obs)
T,E,jj=A.learning_HM(obs,1000)
print('transition: \n')
print(T)
print('\n')
print('Emession: \n')
print(E)
print(A.transition_probabilities)
print('\n')
print('sequence: \n')
print(obs)
print('\n')
print(list(seq))




