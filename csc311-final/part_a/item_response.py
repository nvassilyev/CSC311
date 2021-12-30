from numpy.lib.function_base import diff
from scipy import sparse
from utils import *
import matplotlib.pyplot as plt

import numpy as np

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))
    # return .5 * (1 + np.tanh(.5 * x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # ones = np.ones((np.shape(data)))
    # diff = np.multiply(ones.T, theta).T - np.multiply(ones, beta)
    # log_lklihood = np.sum(np.multiply(data, diff) - np.logaddexp(0, diff))
    
    log_lklihood = 0
    for i in range(len(data["user_id"])):
        s = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        log_lklihood += c * (theta[s] - beta[q]) - np.logaddexp(0, theta[s] - beta[q])
    
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    
    df_theta = np.zeros(542)
    df_beta = np.zeros(1774)

    for i in range(len(data["user_id"])):
        s = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        
        df_theta[s] += c - sigmoid(theta[s]-beta[q])
        df_beta[q] += -c + sigmoid(theta[s]-beta[q])
        
    theta = theta + lr * df_theta
    beta = beta + lr * df_beta
    
    #ones = np.ones((np.shape(data)))
    # diff = np.multiply(ones.T, theta).T - np.multiply(ones, beta) 
       
    # df_theta = lr * np.sum(data - sigmoid(diff), axis = 1)
    # df_beta = lr * np.sum(-data + sigmoid(diff), axis = 0)
    # print(data - sigmoid(diff))
    
    # theta = theta.reshape(542,1) - df_theta
    # beta = beta.reshape(1774,1) - df_beta.reshape(1774,1)
        
    # theta2 = np.zeros(np.shape(data)[0])
    # beta2 = np.zeros(np.shape(data)[1])
    # for i in range(542):
    #     theta2[i] = theta[i]
    # for i in range(1774):
    #     beta2[i] = beta[i]
        
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)
    # theta = np.random.rand(np.shape(data)[0])
    # beta = np.random.rand(np.shape(data)[1])
    

    val_acc_lst = []
    neg_lld_lst1 = []
    neg_lld_lst2= []

    for i in range(iterations):
        neg_lld1 = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_lst1.append(neg_lld1)
        neg_lld2 = neg_log_likelihood(val_data, theta=theta, beta=beta)
        neg_lld_lst2.append(neg_lld2)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld1, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        print(score)

    return theta, beta, val_acc_lst, neg_lld_lst1, neg_lld_lst2


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
        
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])
           


def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    
    sparse_matrix = sparse_matrix.todense()
    sparse_matrix = np.nan_to_num(sparse_matrix)
    lr = 0.01
    iterations = 30
    container = irt(train_data, val_data, lr, iterations)
    
    # i = [i for i in range(iterations)]
    # plt.plot(i, container[3])
    # plt.plot(i, container[4])
    # plt.xlabel("# of iterations")
    # plt.ylabel("neg log-llkihood")
    # plt.legend(["training NLLK", "validation NLLK"])
    # plt.title("NLLK vs. # of iterations")
    # plt.show()
    # 0.7067456957380751 - validation
    # 0.7053344623200677 - test
    
    theta, beta = np.sort(container[0]), container[1]
    j1, j2, j3 = 4, 7, 9
    prob1 = np.sort(sigmoid(theta - beta[j1]))
    prob2 = np.sort(sigmoid(theta - beta[j2]))
    prob3 = np.sort(sigmoid(theta - beta[j3]))

    plt.plot(theta, prob1)
    plt.plot(theta, prob2)
    plt.plot(theta, prob3)
    plt.xlabel("theta values")
    plt.ylabel("probability of correct response")
    plt.legend(["j1", "j2", "j3"])
    plt.title("Probability of Correct Reponses vs Theta values")
    plt.show()
   


if __name__ == "__main__":
    main()
