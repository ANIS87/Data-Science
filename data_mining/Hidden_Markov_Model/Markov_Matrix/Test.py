
import Transistion_Matrix_Extraction as Markov_Class

print ('....test...')

features=['track_id','created_at','browser','src','referrer','event' ]
id_user='track54f7fd26841360.17391116'
data=Markov_Class.matrix_transition('data_brut_clustering.csv',features,10**6)
data.get_all_states(id_user)
transition_matrix_dic,transition_matrix_array=data.visualize_matrix(id_user)
print(transition_matrix_array)
print (data.data)


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


