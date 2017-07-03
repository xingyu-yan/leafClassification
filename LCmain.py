'''
Created on December 3, 2016
@author: xingyu, at Ecole Centrale de Lille
# This programme is for PV power forecasting with ANN
# reference: Coursera Machine Learning open course (Andrew Ng)
# reference: https://www.kaggle.com/c/leaf-classification
'''
 
import pandas as pd  #pandas: Python Data Analysis Library
from numpy import sort
import numpy as np
import scipy.io as sio
from lc_package.LCfuncs import nnCostFunction, sigmoidGradient, randInitializeWeights, checkNNGradients, cgbt, predict
from lc_package.LCfuncs import print_results

print("Loading Data ...\n")

'''csvdf  = pd.read_csv('train.csv', sep=',')
example1 = csvdf[ (csvdf.id < 6) & (csvdf.margin1 == 0) ]
print(example1)
speciesCol = csvdf.ix[:,['species']]
print(speciesCol.describe())
data = csvdf
y = data.pop('species')
print(y.shape)
datasort = sort(speciesCol)'''

myData = sio.loadmat('DataTrain.mat')

X_train = myData['X_train']
y_train = myData['y_train']

input_layer_size = 192   
hidden_layer_size = 25  
num_labels = 99

# Part 2: Loading parameters

mat_contents = sio.loadmat('Theta.mat')

Theta1 = mat_contents['Theta1']
Theta2 = mat_contents['Theta2']

nn_params1 = np.matrix(np.reshape(Theta1, Theta1.shape[0]*Theta1.shape[1], order='F')).T
nn_params2 = np.matrix(np.reshape(Theta2, Theta2.shape[0]*Theta2.shape[1], order='F')).T
nn_params = np.r_[nn_params1,nn_params2]

# Part 3: Compute cost (feedforward)
lamb = 0
J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at parameters without regulariyation (loaded from Theta):',J)

# Part 4: Implement regularization
lamb = 1
J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at parameters with regularization (loaded from Theta):',J)

# Part 7: Implement backpropagation
checkNNGradients(0)

# Part 8: Implement regularization
lamb = 0.3
checkNNGradients(lamb)
debug_J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at (fixed) debugging parameters (lambda =',lamb,'):',debug_J[0],'\n')

# Part 5: Sigmoid gradient
g = sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
print ('Evaluation sigmoid gradient\n',g,'\n')

# Part 6: Initializing parameters

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)

initial_nn_params = np.r_[np.reshape(initial_Theta1,Theta1.shape[0]*Theta1.shape[1],order='F'),np.reshape(initial_Theta2,Theta2.shape[0]*Theta2.shape[1],order='F')]

# Part 9: Training NN
lamb = 0.01
#Theta = cgbt(initial_nn_params,X,y,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,500,1e-8)
Theta = cgbt(initial_nn_params,X_train,y_train,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,10,1e-8)

Theta1 = np.matrix(np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
Theta2 = np.matrix(np.reshape(Theta[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))

# Part 11: Implement predict
p = predict(Theta1,Theta2,X_train)
precision = 0
for i in range(len(y_train)):
    if y_train[i] == p[i]:
        precision += 1

print ('Training Set Accuracy:',(1.0*precision)/len(y_train))