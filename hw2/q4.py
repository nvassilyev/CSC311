# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg, newaxis
from numpy.core.fromnumeric import shape
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    
    test_datum = test_datum[np.newaxis, :]
    N = np.shape(x_train)[0]
    norms = l2(x_train, test_datum)
    
    num = np.exp(-norms/(2*(tau**2)))
    den = np.exp(logsumexp(-norms/(2*(tau**2))))
    
    A = np.zeros((N, N))  
    np.fill_diagonal(A, num / den)
    w = linalg.solve((x_train.T @ A @ x_train + lam *\
           np.identity(d)), x_train.T @ A @ y_train)
    
    return test_datum @ w

    
def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of validation losses, one for each tau value
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    validation_losses = []
    
    for tau in taus:
       loss = 0
       for i in range(np.shape(x_test)[0]):
          y_hat = LRLS(x_test[i], x_train, y_train, tau)
          loss += (y_test[i] - y_hat) ** 2
          
       validation_losses.append(loss/np.shape(x_train)[0])
       print(loss/np.shape(x_train)[0])
    return validation_losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,4,200)
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.xlabel("tau")
    plt.ylabel("validation loss")
    plt.title("Validation losses as a function of tau values")
    plt.semilogx(taus, test_losses)
    plt.show()

