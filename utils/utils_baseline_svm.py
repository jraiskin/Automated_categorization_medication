import numpy as np
from collections import Counter

# creates a list of arrays
# each have an entry of 1.0 where that class is defined (in the classes array)
def y_to_dense(y, classes, dtype=float):
    return [(y[i] == classes).astype(dtype) 
            for i in range(len(y))]


# get the ranking of the elements in an array (argmax is 1)
# order is maintained
def get_rank_order(input_array):
    temp = input_array.argsort()
    ranks = np.empty(len(input_array), int)
    ranks[temp] = np.arange(len(input_array))
    return np.shape(ranks)[0] - ranks


# return a the % of observations with the correct label at the top k
def in_top_k(y_dense, log_pred, k):
    assert len(y_dense) == len(log_pred), 'y and predictions are not of same length'
    return np.mean(
        [get_rank_order(log_pred[i])[np.argmax(y_dense[i])] <= k
         for i in range(len(y_dense))])


# returns the mean reciprocal rank
def mean_reciprocal_rank(y_dense, log_pred):
    assert len(y_dense) == len(log_pred), 'y and predictions are not of same length'
    return np.mean(
        [1.0 / get_rank_order(log_pred[i])[np.argmax(y_dense[i])]
         for i in range(len(y_dense))])


if __name__ == '__main__':
    pass
