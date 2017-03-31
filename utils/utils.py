from getpass import getuser
# getpass.getuser()
# 'yarden'

import pickle
#import re
import inspect
import pandas as pd
import numpy as np
from math import ceil

from collections import Counter

import random


def seed():
    return 2178


# set seeds
np.random.seed(seed())
random.seed(seed())


def user_opt_gen():
    user_opt = {
        'yarden' : {
            'data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/20170303_EXPORT_for_Yarden.csv',
    #         'data_path' : r'/home/yarden/git/Automated_categorization_medication/data/20170303_EXPORT_for_Yarden.csv',
            'atc_conversion_data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/Complete_ATCs_and_lacking_translations_V03a_20161206.csv', 
#            'suggested_labels' : r'/home/yarden/git/Automated_categorization_medication/data/20170303_EXPORT_for_Yarden.csv'
            'suggested_labels' : r'/home/yarden/git/Automated_categorization_medication/similarity_labels_suggestion.csv'
        },
        'raiskiny' : {
            'data_path' : r'/cluster/home/raiskiny/thesis_code_and_data/data/20170303_EXPORT_for_Yarden.csv', 
            'atc_conversion_data_path' : r'/cluster/home/raiskiny/thesis_code_and_data/data/Complete_ATCs_and_lacking_translations_V03a_20161206.csv',
            'suggested_labels' : r'/cluster/home/raiskiny/thesis_code_and_data/data/similarity_labels_suggestion.csv'
        },
        'Yarden-' : {
            'data_path' : None, 
            'atc_conversion_data_path' : None
        }
    }
    cur_user = getuser()
    return user_opt[cur_user]


def init_data():
    user_opt = user_opt_gen()

    main_data = pd.read_csv(user_opt['data_path'], 
                            sep=';', 
                            header=0, 
                            encoding='cp850')

    # only observations with ATC labels
    main_data_labeled = main_data.loc[[isinstance(k, str) for k in main_data['ATC']],:]
    
    n = len(main_data_labeled)

    x = main_data_labeled['FREETXT'][:n]
    y = main_data_labeled['ATC'][:n]
    y = [i for i in y]  # make into a list
    
    return x, y, n, main_data


# initialize data from suggestions CSV file
def init_data_suggest():
    _, _, _, main_data = init_data()
    
    user_opt = user_opt_gen()
    suggested_data = pd.read_csv(user_opt['suggested_labels'], 
                                 sep=',', 
                                 names=['Text', 'ATC', 'Jaccard_sim'], 
                                 encoding='cp850')
    
    x_suggest = [i for i in suggested_data['Text']]
    y_suggest = [i for i in suggested_data['ATC']]
    
    main_freq = main_data['CNT']
    main_text = main_data['FREETXT']
    
    # match frequency from main file with text from suggestions CSV file
    freq_suggest = [main_freq[[text == t for t in main_text]] for text in x_suggest]
    # extract values and convert to int (instead of np.int64)
    freq_suggest = [int(i.values[0]) for i in freq_suggest]
    # n_suggest = len(freq_suggest)
    
    # check that lengths match
    cond = len(freq_suggest) == len(x_suggest)
    # check that all are int
    cond = cond and not any(x for x in freq_suggest if not isinstance(x, int))
    # check that there are unexpected values
    cond = cond and not any(x for x in freq_suggest if x <= 0)
    cond = cond and not any(x for x in freq_suggest if x > 100)
    cond = cond and not any(x for x in freq_suggest if x == [])
    # also performed some manual inspection, comparing against the CSV files
    # list(zip(x_suggest, freq_suggest))[:10]
    assert cond, 'The variable "freq_suggest" failed one of the completeness tests.'
    
    return x_suggest, y_suggest, freq_suggest


def train_validation_split(*, x, y, freq, label_count_thresh, valid_ratio, verbose=True):
    # count each label occurence, filter out those less frequent than label_count_thresh
    label_freq_dict = Counter(y)
    label_freq_dict = {label:count for label,count in label_freq_dict.items() if count >= label_count_thresh}

    # create a dict with a list of label:list(index), for filtered labels
    y_enum = [(i,label) for (i,label) in enumerate(y) if label in label_freq_dict.keys()]
    label_index_dict = {label:[] for label in label_freq_dict.keys()}
    for (i,label) in y_enum:
        label_index_dict[label].append(i)

    # check
    assert(min([len(x) for x in label_index_dict.values()]) >= label_count_thresh)
    
    label_valid_index_dict = {label:random.sample(indices, int(valid_ratio * len(indices)))
                              for label,indices in sorted(label_index_dict.items())}
    if verbose:
        n_val = sum([len(x) for x in label_valid_index_dict.values()])
        print('number of potential labels to draw from: {}'.format(len(label_index_dict)))
        print('number of potential observation to draw from: {}'.format(
            sum([len(v) for v in label_index_dict.values()])))
        print('{} observations sampled for validation'.format(n_val))
        print('{} total number of observations'.format(len(y)))
        print('{} observations for training'.format(len(y) - n_val))
        print('The ratio of validation to total observations is about {:.3f}'.format(n_val / len(y)))

    # extract all indices into a single set
    valid_index = [item for sublist in list(label_valid_index_dict.values()) for item in sublist]
    valid_index.sort()

    # construct 2 sets of variables, validation and training
    x_val, x_train, y_val, y_train, freq_val, freq_train = [], [], [], [], [], []
    for i, (x, y, freq) in enumerate(zip(x, y, freq)):
        if i in valid_index:
            x_val.append(x)
            y_val.append(y)
            freq_val.append(freq)
        else:
            x_train.append(x)
            y_train.append(y)
            freq_train.append(freq)
    
    return x_val, x_train, y_val, y_train, freq_val, freq_train, valid_index


def unscale(freq):
    return int(1)


def log_scale(freq, shift = 0.0, scale = 1.0):
    eps = 0.001
    return int(ceil(np.log10((freq + shift + eps) / scale)
                    ))


def replicate_dict(x, y, freq, scale_func=unscale):
    row_replicate_dict = {}
    for ind, (obs, label, freq) in enumerate(zip(x, y, freq), start=1):
    #     print(ind, obs, label, freq)
        row_replicate_dict[ind] = (np.array([obs, ] * scale_func(freq)), 
                                   label)
        
    return row_replicate_dict


def scale_permute_data(*, x, y, freq, scale_func, to_permute=True):
    row_replicate_dict = replicate_dict(x, y, freq, scale_func)
    
    # retreive character sequences (element 0 of tuple)
    x_scaled = np.concatenate([row_replicate_dict[i][0] 
                               for i in range(1, len(row_replicate_dict) + 1)], 
                              axis=0)
    
    # retreive label sequence (element 1 of tuple)
    # labels are replicated to match the scaled version
    # of the character sequence
    y_scaled = np.concatenate([np.array([row_replicate_dict[i][1],] * 
                                        row_replicate_dict[i][0].shape[0])
                               for i in range(1, len(row_replicate_dict) + 1)], 
                              axis=0)
    
    # checks that no funky business is going on
    n = sum([scale_func(num) for num in freq])
    cond = x_scaled.shape[0] == n
    cond = cond and y_scaled.shape == (n, )
    assert cond, 'There we unexpected shapes in either of "x_scaled" or "y_scaled" variables. Please check!'
    
    if to_permute:
        permute = np.random.permutation(n)
        x_scaled = x_scaled[permute]
        y_scaled = y_scaled[permute]
    
    return x_scaled, y_scaled, n


def save(fname, obj):
    with open(fname, 'w') as f:
        pickle.dump(obj, f)


def load(fname):
    with open(fname, 'r') as f:
        return pickle.load(f)


# dict with a more convenient way of calling (key, val)
class nice_dict(dict):

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __delattr__(self, key):
        del self[key]


def print_source(function):
    return print(inspect.getsource(function))


# let's easily follow work-flow, use with 'with'
class in_out:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print("\033[1;33m---enter {}---\033[0m".format(self.name))
        return self
    def __exit__(self, *args):
        print("\033[1;33m---exit {}---\033[0m".format(self.name))


def pcp1():
    return print("\033[1;32m---checkpoint 1---\033[0m")


def pcp2():
    return print("\033[1;33m---checkpoint 2---\033[0m")


def pcp3():
    return print("\033[1;34m---checkpoint 3---\033[0m")


def pcp4():
    return print("\033[1;35m---checkpoint 4---\033[0m")


if __name__ == '__main__':
    print("\033[1;32m---checkpoint 1---\033[0m")
    print("\033[1;33m---checkpoint 2---\033[0m")
    print("\033[1;34m---checkpoint 3---\033[0m")
    print("\033[1;35m---checkpoint 4---\033[0m")
    print("\033[1;36m---checkpoint 5---\033[0m")
    print("\033[1;37m---checkpoint 6---\033[0m")
    print("\033[1;38m---checkpoint 7---\033[0m")
    print("\033[1;39m---checkpoint 8---\033[0m")
    print("\033[1;40m---checkpoint 9---\033[0m")
    #    cur_user = getpass.getuser()
    #    user_opt = user_opt[cur_user]

