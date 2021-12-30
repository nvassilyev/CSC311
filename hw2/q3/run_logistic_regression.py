from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,
        "num_iterations": 1000
    }

    weights = np.zeros((M+1, 1))
    
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    ce = [[],[],[]]
    error = [[],[],[]]
    
    for _ in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - hyperparameters["learning_rate"] * df
        ce_train, correct_train = evaluate(train_targets, logistic_predict(weights, train_inputs))
        ce_valid, correct_valid = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
        ce_test, correct_test = evaluate(test_targets, logistic_predict(weights, test_inputs))
        
        ce[0].append(ce_train)
        ce[1].append(ce_valid)
        ce[2].append(ce_test)
        error[0].append(1 - correct_train)
        error[1].append(1 - correct_valid)
        error[2].append(1 - correct_test)
        
        
    return ce, error

def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    ce, error = run_logistic_regression()
    x = [i for i in range(1000)]
    plt.plot(x, ce[0])
    plt.plot(x, ce[1])
    plt.legend(["training", "validation"])
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("CE loss as a function of iterations")
    plt.show()
    print(f'Validation: Min CE: {min(ce[1])} with index {ce[1].index(min(ce[1]))}, error: {error[1][ce[1].index(min(ce[1]))]}')
    print(f'Training: Min Ce: {ce[0][999]} error: {error[0][999]}')
    print(f'Testing: Min Ce: {ce[2][999]} error: {error[2][999]}')
