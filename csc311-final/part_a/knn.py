from sklearn.impute import KNNImputer
from utils import *
import matplotlib .pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    load_train_sparse_with_meta("../../../PycharmProjects/temp/data")
    sparse_matrix = load_train_sparse("../../../PycharmProjects/temp/data").toarray()
    val_data = load_valid_csv("../../../PycharmProjects/temp/data")
    test_data = load_public_test_csv("../../../PycharmProjects/temp/data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k = [1, 6, 11, 16, 21, 26]
    user_validation_accuracy = []
    item_validation_accuracy = []
    user_test_accuracy = []
    item_test_accuracy = []
    for i in k:
        print("KNN by user: {}".format(i))
        # user_validation_accuracy.append(knn_impute_by_user(sparse_matrix, val_data, i))
        user_test_accuracy.append(knn_impute_by_user(sparse_matrix, test_data, i))
        print("KNN by item: {}".format(i))
        # item_validation_accuracy.append(knn_impute_by_item(sparse_matrix, val_data, i))
        item_test_accuracy.append(knn_impute_by_item(sparse_matrix, test_data, i))

        print("\n")

    # plot the accuracy
    # plt.plot(k, user_validation_accuracy, label="Validation Accuracy")
    plt.plot(k, item_validation_accuracy, label="Validation Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
