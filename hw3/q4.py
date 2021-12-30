'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import math
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for c in range(10):
        class_set = train_data[np.where(train_labels == c)]
        means[c,:] = np.mean(class_set, axis=0)
    
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for c in range(10):
        class_set = train_data[np.where(train_labels == c)]
        covariances[c,:] = (class_set-means[c,:]).T @ (class_set-means[c,:]) + 0.01
        
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    log_like = np.zeros((np.shape(digits)[0], 10))
    
    for c in range(10):
        log_c = np.log((2*math.pi)**(-32) * np.linalg.det(covariances[c])**(-0.5) *\
            (-0.5 * (digits - means[c]).T @ np.linalg.inv(covariances[c]) @\
                (digits - means[c])))
        log_like[:,c] = log_c
        
    return log_like

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    log_like = np.zeros((np.shape(digits)[0], 10))
    gen_logs = generative_likelihood(digits, means, covariances)
    
    
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    
    # Leading Eigenvectors
    eigen = np.zeros((10,8,8))
    for c in range(10):
        a,b = np.linalg.eigh(covariances[c])
        eigen[c, :] = np.reshape(b[np.argmax(a)], (8,8))
    
    figure, axis = plt.subplots(2, 5)
    
    for i in range(2):
        for j in range(5):
            axis[i,j].imshow(eigen[i*5 + j], cmap='gray')
            axis[i,j].set_title(f'Digit: {i*5 +j}')
    plt.show()
    
    
    

if __name__ == '__main__':
    main()
