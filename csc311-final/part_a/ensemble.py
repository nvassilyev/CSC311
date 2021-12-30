from utils import *
import numpy as np

from neural_network import *
from sklearn.impute import KNNImputer


def load_bootstrap_sparse(n, base_path="../data"):
    """
    Loads the training matrix, then runs bootstrapping and returns the bootsrapped matrix.
    n: number of samples
    base_path: path to the data directory
    """
    train_matrix = load_train_sparse(base_path)

    num_users = train_matrix.shape[0]

    # generate an array of n random variables between 0 and num_users
    indices = np.random.randint(0, num_users, n)

    # create the bootstrap matrix
    bootstrap_matrix = train_matrix[indices, :]

    return bootstrap_matrix


def nn_bootstrap_data(bootstrap):
    """
    The same as load_data function in neural_network.py, but you provide the 
    training sparse matrix.

    bootstrap: the training matrix
    """
    base_path = '../data'

    train_matrix = bootstrap.toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


if __name__ == "__main__":
    n = 2000

    train_matrix = load_bootstrap_sparse(n, "../data")
    knn_train = train_matrix.copy()

    # Load nn data.
    nn_zero_train_matrix, nn_train_matrix, valid_data, test_data = nn_bootstrap_data(
        train_matrix)

    # Neural Network
    lr1 = 0.001
    num_epoch1 = 200
    lamb1 = 0  # dont want regularization right now
    k1 = 200

    nn1 = AutoEncoder(train_matrix.shape[1], k=k1)

    lr2 = 0.001
    num_epoch2 = 200
    lamb2 = 0  # dont want regularization right now
    k2 = 200

    nn2 = AutoEncoder(train_matrix.shape[1], k=k2)

    # train(nn1, lr1, lamb1, train_matrix,
    #       nn_zero_train_matrix, nn_valid_data, num_epoch1)
    # train(nn2, lr2, lamb2, train_matrix,
    #       nn_zero_train_matrix, nn_valid_data, num_epoch2)
    

    # TODO other stuff

    knn_neighbours = 1
    knn = KNNImputer(n_neighbors=knn_neighbours)

    # impute by user
    knn_in = knn_train.toarray()
    print('knn in shape ', knn_in.shape)
    knn_out = knn.fit_transform(knn_in)

    print('knn out shape', knn_out.shape)


    # make either validation of testing
    eval_data = valid_data

    # ensembling the models
    correct = 0
    total = 0
    for i, u in enumerate(eval_data["user_id"]):
        # neural network guess
        nn_inputs = Variable(nn_zero_train_matrix[u]).unsqueeze(0)

        nn_output1 = nn1(nn_inputs)
        nn_guess1 = nn_output1[0][eval_data["question_id"][i]].item() >= 0.5

        nn_output2 = nn2(nn_inputs)
        nn_guess2 = nn_output2[0][eval_data["question_id"][i]].item() >= 0.5

        # knn guess
        cur_user_id = eval_data["user_id"][i]
        cur_question_id = eval_data["question_id"][i]

        knn_guess = knn_out[cur_user_id, cur_question_id] >= 0.5

        # take the majority vote of the three models
        majority_guess = nn_guess1 or knn_guess if nn_guess2 else nn_guess1 and knn_guess
        
        total += 1
        if majority_guess == eval_data["is_correct"][i]:
            correct += 1

    print('accuracy: ', correct / float(total))

