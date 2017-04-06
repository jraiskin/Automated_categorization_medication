import tensorflow as tf
import numpy as np
import os

try:
    from utils.utils_baseline_svm import filter_dict_by_val_atleast, char_freq_map
except:
    from utils_baseline_svm import filter_dict_by_val_atleast, char_freq_map
# try:
#     from utils.utils import seed
# except:
#     from utils import seed



# set seeds
# np.random.seed(seed())
# random.seed(seed())


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


def text_filter_pad(text, y, char_filter, filter_keys_chars=None, *args, **kwagrs):
    """
    Filter characters, leaving only those that appear at least 'char_filter' times in the text.
    Can also except a pre-defined filter if 'filter_keys_chars' is given (default is None.
    Replaces unknown characters with an "unknown" symbol.
    Pads each line, so that all lines are the same length
    """
    if filter_keys_chars is None:
        filter_keys_chars = list(
            filter_dict_by_val_atleast(
                input_dict=char_freq_map(input_data=text), 
                value=char_filter)
            .keys())    
    else:
        filter_keys_chars = filter_keys_chars
    
    # create a list of character lists
    x_char = [list(line) for line in text]
    x_char_filtered = []
    unknown = '<unk-char>'
    # replace chars not in 'filter_keys_chars' with 'unknown'
    for line in x_char:
        x_char_filtered.append([char if (char in filter_keys_chars) else unknown for char in line])
    
    # pad lines, so that all lines are the same length
    max_line_len = int(np.max([len(line) for line in text]))
    pad = '<pad-char>'
    x_char_filtered_pad = []
    for i, line in enumerate(x_char_filtered):
        x_char_filtered_pad.append(pad_list(input_list=line, 
                                        max_length=max_line_len, 
                                        pad_symbol=pad))
    
    # additional statistics based on filtered features
    label_set = set(y)
    n_label = len(label_set)
    # number of unique characters in input ('x_char_filtered')
    char_set = set([char for line in x_char_filtered_pad for char in line])
    n_char = len(char_set)
    statistics_dict={'seq_len': max_line_len,
                     'n_class': n_label,
                     'n_char': n_char,
                     'char_set': char_set, 
                     'label_set': label_set, 
                     'filter_keys_chars': filter_keys_chars
                    }
    
    return x_char_filtered_pad, statistics_dict


def lookup_dicts_chars_labels(char_set, label_set, *args, **kwagrs):
    """
    Create the following dictionaries:
    {character: int} and its inverse {int: character},
    {label: int} and its inverse {int: label}
    """
    # create lookup dict for characters (and inv)
    char_int = {}
    char_int_inv = {}
    char_set = sorted(list(char_set))
    for i, char in enumerate(char_set):
        char_int[char] = i
        char_int_inv[i] = char

    # same for labels
    label_int = {}
    label_int_inv = {}
    label_set = sorted(list(label_set))
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


def assert_no_stats_change(new_dict, kwargs):
    # check that there are no "new" statistics popping out
    # label_set and n_class should not be expected to be the same
    # (and are indeed not, but it is a subset of the original label set, by definition)
    diff_stat = []
    for key in new_dict:
        if not new_dict[key] == kwargs[key]:
            diff_stat.append(key)
    #         print('Statistics differ on {}'.format(key))
    assert set(diff_stat) == {'label_set', 'n_class'}, \
        'Found unexpected values between original x and x_suggest!\n' + \
        'Differences found here: {}\n'.format(diff_stat) + \
        '(label_set and n_class should not be expected to be the same.)'


def make_hparam_string(*, learn_rate, 
                          dynamic_learn_rate, 
                          bidirection, 
                          one_hot, 
                          keep_prob, 
                          char_embed_dim, 
                          hidden_state_size, 
                          scale_func, 
                          keep_rare_labels, 
                          l2_wieght_reg, 
                          target_rep,
                          target_rep_weight,
                          **kwargs):
    scale_func_str = 'scale_func={}'.format(scale_func.__name__)
    bidirection_str = 'bidirection={:.1}'.format(str(bidirection))
    keep_rare_labels_str = 'keep_rare_labels={:.1}'.format(str(keep_rare_labels))
    if dynamic_learn_rate:
        learn_rate_str = 'learn_rate=dynamic'
    else:
        learn_rate_str = 'learn_rate={:.1E}'.format(learn_rate)
    # one_hot_str = 'one_hot={:.1}'.format(str(one_hot))
    keep_prob_str = 'keep_prob={:.2}'.format(keep_prob)
    if one_hot:
        char_embed_dim_str = 'one_hot'
    else:
        char_embed_dim_str = 'char_embed_dim={}'.format(char_embed_dim)
    hidden_state_size_str = 'hidden_state_size={}'.format(hidden_state_size)
    l2_wieght_reg_str = 'l2_wieght_reg={:.1E}'.format(l2_wieght_reg)
    target_rep_weight_str = 'target_rep_weight={}'.format(target_rep_weight if target_rep else 'NA')
    output_str = ",".join([scale_func_str, 
                           bidirection_str, 
                           keep_rare_labels_str, 
                           learn_rate_str, 
                           keep_prob_str, 
                           char_embed_dim_str, 
                           hidden_state_size_str, 
                           l2_wieght_reg_str, 
                           target_rep_weight_str])
    return '{}/'.format(output_str)


def write_embeddings_metadata(log_dir, dictionary, file_name='metadata.tsv'):
    embed_vis_path = './{}{}'.format(log_dir, file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file = open(embed_vis_path,'w')
    file.write('Index\tCharacter')  # tab seperated
    for k, v in sorted(dictionary.items()):
        file.write('\n{}\t{}'.format(v, k))
    file.close()
    
    return embed_vis_path
    
