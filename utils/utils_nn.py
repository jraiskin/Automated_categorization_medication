import tensorflow as tf
import numpy as np

from utils.utils_baseline_svm import filter_dict_by_val_atleast, char_freq_map

def remove_dir_content(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
        print('Log directory was deleted.')
    else:
        print('Log directory was not found.')
#         print(path)


# pad a list to max_length with the pad_symbol
def pad_list(*, input_list, max_length, pad_symbol):
    output_list = input_list + [pad_symbol] * (max_length - len(input_list))
    return output_list


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    

def index_to_dense(index, length):
    output_list = [0.0] * length
    output_list[index] = 1.0
    return output_list


def text_filter_pad_to_index(text, y, char_filter, *args, **kwagrs):
    # filter by character, appear at least 'char_filter' times in the input
    filter_keys_chars = list(
        filter_dict_by_val_atleast(
            input_dict=char_freq_map(input_data=text), 
            value=char_filter)
        .keys())    
    
    # create a list of character lists
    x_char = [list(line) for line in text]
    x_char_filtered = []
    unknown = '<unk-char>'
    # replace chars not in 'filter_keys_chars' with 'unknown'
    for line in x_char:
        x_char_filtered.append([char if (char in filter_keys_chars) else unknown for char in line])
    
    # pad lines, so that all lines are same length
    max_line_len = int(np.max([len(line) for line in text]))
    pad = '<pad-char>'
    x_char_filtered_pad = []
    for i, line in enumerate(x_char_filtered):
        x_char_filtered_pad.append(pad_list(input_list=line, 
                                        max_length=max_line_len, 
                                        pad_symbol=pad))
    
    # additional statistics based on filtered features
    label_set = y.unique()
    n_label = len(label_set)
    # number of unique characters iin input ('x_char_filtered')
    char_set = set([char for line in x_char_filtered_pad for char in line])
    n_char = len(char_set)
    statistics_dict={'seq_len': max_line_len,
                     'n_class': n_label,
                     'n_char': n_char,
                     'char_set': char_set, 
                     'label_set': label_set
                    }
    
    return x_char_filtered_pad, statistics_dict


def lookup_dicts_chars_labels(char_set, label_set, *args, **kwagrs):
    # create lookup dict for characters (and inv)
    char_int = {}
    char_int_inv = {}
    for i, char in enumerate(char_set):
        char_int[char] = i
        char_int_inv[i] = char

    # same for labels
    label_int = {}
    label_int_inv = {}
    for i, label in enumerate(label_set):
        label_int[label] = i
        label_int_inv[i] = label
    
    return char_int, char_int_inv, label_int, label_int_inv


def index_transorm_xy(x, 
                      y, 
                      char_int, 
                      label_int, 
                      n_class, 
                      *args, **kwagrs):
    # transform x from a list of symbols into a list of ints
    X = []
    for line in x:
        X.append([char_int[char] for char in line])
    
    # create Y as a list of list(int)
    Y = [[label_int[label]] for label in y]
    
    # transform into format acceptable by tf
    X = np.array(X)
    Y_dense = np.array(
        [index_to_dense(label[0], 
                        n_class) for label in Y])
    
    return X, Y, Y_dense