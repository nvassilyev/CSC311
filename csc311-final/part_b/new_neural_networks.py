from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

from torchviz import make_dot

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.hidden = nn.Linear(k, k) # our customization is a hidden layer
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        hidden_w_norm = torch.norm(self.hidden.weight, 2) ** 2  #our custom layer
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        g_output = self.g(inputs)
        g_output = F.relu(g_output)

        hidden_output = self.hidden(g_output)
        hidden_output = F.relu(hidden_output)

        h_output = self.h(hidden_output)
        h_output = F.sigmoid(h_output)
        out = h_output

        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: a list of validation accuracies for each epoch; and a list of training loss for each epoch.
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    val_accs = []
    train_losses = []

    for epoch in tqdm(range(0, num_epoch)):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += model.get_weight_norm() * lamb
            
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        val_accs.append(valid_acc)
        train_losses.append(train_loss)
        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #   "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return val_accs, train_losses


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)



def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    #                                                                   #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 200

    k_list = [10, 50, 100, 200, 500]
    k = 200
    lamb=0

    epochs_list = range(1, num_epoch+1)

    
    # ! THIS PLOTS DIFFERENT K VALUES AS FUNCTION OF EPOCH

    plt.title(f'Custom Neural Network (lr={lr}; k={k})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')



    for k in k_list:
        print(f'Training k={k}')

        model = AutoEncoder(train_matrix.shape[1], k=k)

        val_accs = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        plt.plot(epochs_list, val_accs, label=f'k={k}')

    

    # # ! THIS IS FOR WHEN YOU WANT TO TEST THE END RESULT AFTER PICKING HYPERPARAMS
    # model = AutoEncoder(train_matrix.shape[1], k=k)

    # # commented unless you want diagram
    # dummy_x = Variable(zero_train_matrix[0]).unsqueeze(0).requires_grad_(True)
    # yhat = model(dummy_x)
    # make_dot(yhat, params=dict(list(model.named_parameters())+[('x', dummy_x)])).render("partB-nn", format="png")
    
    # print(f'Training lr={lr} epochs={num_epoch} k={k} lamb={lamb}')

    # val_accs, train_losses = train(model, lr, lamb, train_matrix,
    #                  zero_train_matrix, valid_data, num_epoch)

    # print(f'Results (lr={lr} epochs={num_epoch} k={k} lamb={lamb})')
    # print('Validation: ', evaluate(model, zero_train_matrix, valid_data))
    # print('Test:\t\t ', evaluate(model, zero_train_matrix, test_data))


    # plt.title(f'Validation Accuracy (lr={lr} epochs={num_epoch} k={k} lamb={lamb})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy')
    # plt.plot(epochs_list, val_accs)
    # plt.show()

    # plt.title(f'Training Loss (lr={lr} epochs={num_epoch} k={k} lamb={lamb})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.plot(epochs_list, train_losses)
    # plt.show()


# plt.legend()
# plt.show()

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################


if __name__ == "__main__":
    main()
