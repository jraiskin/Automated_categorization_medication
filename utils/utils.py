from getpass import getuser

import pickle
#~ import re
#~ import inspect
import pandas as pd
import numpy as np
from math import ceil
import csv

from collections import Counter

import random


def seed():
    return 2178


def user_opt_gen():
    user_opt = {
        'yarden' : {
            'data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/20170303_EXPORT_for_Yarden.csv',
    #         'data_path' : r'/home/yarden/git/Automated_categorization_medication/data/20170303_EXPORT_for_Yarden.csv',
            'atc_conversion_data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/Complete_ATCs_and_lacking_translations_V03a_20161206.csv', 
#            'suggested_labels' : r'/home/yarden/git/Automated_categorization_medication/data/20170303_EXPORT_for_Yarden.csv'
            'suggested_labels_090' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/similarity_labels_suggestion_revised_090_internal_data_filtered.csv', 
            'suggested_labels_080' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/similarity_labels_suggestion_revised_080_internal_data_filtered.csv', 
            'suggested_labels_070' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/similarity_labels_suggestion_revised_070_internal_data_filtered.csv', 
            'wiki_atc_code' : r'/home/yarden/git/Automated_categorization_medication/resources/wiki_scrape_filter.csv', 
            'drugbank_atc_code' : r'/home/yarden/git/Automated_categorization_medication/resources/drugbank_filter.csv'
        },
        'raiskiny' : {
            'data_path' : r'/cluster/home/raiskiny/thesis_code_and_data/data/20170303_EXPORT_for_Yarden.csv', 
            'atc_conversion_data_path' : r'/cluster/home/raiskiny/thesis_code_and_data/data/Complete_ATCs_and_lacking_translations_V03a_20161206.csv',
            'suggested_labels_090' : r'/cluster/home/raiskiny/thesis_code_and_data/data/similarity_labels_suggestion_revised_090_internal_data_filtered.csv', 
            'suggested_labels_080' : r'/cluster/home/raiskiny/thesis_code_and_data/data/similarity_labels_suggestion_revised_080_internal_data_filtered.csv', 
            'suggested_labels_070' : r'/cluster/home/raiskiny/thesis_code_and_data/data/similarity_labels_suggestion_revised_070_internal_data_filtered.csv', 
            'wiki_atc_code' : r'/cluster/home/raiskiny/thesis_code_and_data/data/wiki_scrape_filter.csv', 
            'drugbank_atc_code' : r'/cluster/home/raiskiny/thesis_code_and_data/data/drugbank_filter.csv'
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
                            encoding='iso-8859-15')

    # only observations with ATC labels
    main_data_labeled = main_data.loc[[isinstance(k, str) for k in main_data['ATC']],:]
    
    n = len(main_data_labeled)

    x = main_data_labeled['FREETXT'][:n]
    y = main_data_labeled['ATC'][:n]
    y = [i for i in y]  # make into a list
    
    return x, y, n, main_data


def init_data_other_atc(data_key):
    """
    Initializes data_key from external sources, reads from CSV files.
    data_key refers to the key in the user_opt_gen() object.
    data_key can be in ('wiki_atc_code', 'drugbank_atc_code')
    """
    user_opt = user_opt_gen()
    
    if data_key not in ('wiki_atc_code', 
                        'drugbank_atc_code'):
        raise ValueError('Encoutered unknown data_key')
    
    data = pd.read_csv(user_opt[data_key], 
                       sep=',', 
                       header=0, 
                       encoding='iso-8859-15')
    
    if data_key == 'wiki_atc_code':
        x = [elem for elem in data['Name']]
        y = [elem for elem in data['ATC-Code']]
    else:
        x = [elem for elem in data['name']]
        y = [elem for elem in data['atc_codes']]
    return x, y


# initialize data from suggestions CSV file
def init_data_suggest(use_suggestions):
    # handle the different data suggestion options
    if use_suggestions not in {'0.7', '0.8', '0.9', 'F'}:
        raise ValueError("use_suggestions parameter has to be a string '+\
            'with values of either {'0.7', '0.8', '0.9', 'F'}")
    if use_suggestions == '0.7':
        suggestion_data_key = 'suggested_labels_070'
    elif use_suggestions == '0.8':
        suggestion_data_key = 'suggested_labels_080'
    elif use_suggestions == '0.9':
        suggestion_data_key = 'suggested_labels_090'
    
    x, y, n, main_data = init_data()
    
    x = [i for i in x]
    freq = [i for i in main_data['CNT'][:n]]  # frequencies, turned into a list
    
    main_freq = main_data['CNT']
    main_text = main_data['FREETXT']
    
    user_opt = user_opt_gen()
    # if use_suggestions, load suggestion data from CSV file
    # if not, returning empty lists works
    if use_suggestions in {'0.7', '0.8'}:
        print('Using label suggestion data with similarity threshold of {}'.format(use_suggestions))
        suggested_data = pd.read_csv(user_opt[suggestion_data_key], 
                                     sep=',', 
                                     names=['Text', 'ATC', 'Jaccard_sim'], 
                                     encoding='iso-8859-15')
        
        x_suggest = [i for i in suggested_data['Text']]
        y_suggest = [i for i in suggested_data['ATC']]
        
        # match frequency from main file with text from suggestions CSV file
        freq_suggest = [main_freq[[text == t for t in main_text]] for text in x_suggest]
        # extract values and convert to int (instead of np.int64)
        freq_suggest = [int(i.values[0]) for i in freq_suggest]
    else:
        print('Not using label suggestion data')
        x_suggest, y_suggest, freq_suggest = [], [], []
    
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
    
    return x, y, freq, x_suggest, y_suggest, freq_suggest


def make_chars(input, 
               allowed_chars, 
               unknown_char='<unk-char>'):
    
    result = [list(obs) for obs in input]
    result = [[char if char in allowed_chars 
              else unknown_char for char in obs]
             for obs in result]
    return result


def make_ngrams(input, 
                allowed_ngrams, 
                ngram_width, 
                unknown_ngram='<unk-ngram>'):
    return [obs + [ngram
                   if ngram in allowed_ngrams
                   else unknown_ngram
                   for ngram in 
                   join_sliding_window(obs, 
                       ngram_width)]
            for obs in input]


def data_load_preprocess(*args,
                         ngram_filter,
                         allowed_ngrams,
                         ngram_width,
                         valid_ratio,
                         label_count_thresh,
                         allowed_chars,
                         model,
                         mk_ngrams,
                         scale_func,
                         keep_infreq_labels,
                         to_permute,
                         mk_chars,
                         char_filter, 
                         use_suggestions, 
                         linear_counters=True, 
                         output_intermediate=False,
                         **kwargs):
    
    try:
        from utils.utils_nn import lookup_dicts_chars_labels, text_filter_pad
    except:
        from utils_nn import lookup_dicts_chars_labels, text_filter_pad
    
    assert model in ['linear', 'neural'],\
        'Model has to be identified as either linear or neural'

    ### initialize data from MAIN and SUGGESTION CSV files ###
    x, y, freq, x_suggest, y_suggest, freq_suggest =\
        init_data_suggest(use_suggestions)
    
    ### global counter: characters ###
    if mk_chars:
        char_counter = dict_addition([Counter(obs) for obs in x])
        allowed_chars = [key for key,value in char_counter.items() 
                         if value >= char_filter]
        allowed_chars.sort()

        # replacing unknown characters with UNKNOWN symbol
        unknown_char = '<unk-char>'
        # for x
        x_unk = make_chars(x, 
                           allowed_chars=allowed_chars)
        
#         x_unk = [list(obs) for obs in x]
#         x_unk = [[char if char in allowed_chars 
#                   else unknown_char for char in obs]
#                  for obs in x_unk]
        
        # for suggestion x
        x_suggest_unk = make_chars(x_suggest, 
                                   allowed_chars=allowed_chars)
#         x_suggest_unk = [list(obs) for obs in x_suggest]  # same for x_suggest
#         x_suggest_unk = [[char if char in allowed_chars 
#                           else unknown_char for char in obs]
#                          for obs in x_suggest_unk]
    else:
        allowed_chars = list({char for obs in x for char in obs})
        allowed_chars.sort()
        x_unk = x
        x_suggest_unk = x_suggest

    ### global counter: ngrams ###
    # note: AFTER applying 'unknown' to characters
    if model == 'neural':
        pass
    elif mk_ngrams:
        ngram_counter = dict_addition(
            [Counter(join_sliding_window(obs, ngram_width))
             for obs in x_unk])
        allowed_ngrams = [key for key,value in ngram_counter.items() 
                         if value >= ngram_filter]
        allowed_ngrams.sort()  # len(.) 1925
    else:
        ngrams_list = [join_sliding_window(obs, ngram_width) 
                       for obs in x_unk]
        allowed_ngrams = list({ngram for obs in ngrams_list for ngram in obs})
        allowed_ngrams.sort()  # len(.) 17495
    
    # output for data inspection
    if output_intermediate:
        return x_unk, x_suggest_unk, y, y_suggest, freq, freq_suggest
    
    ### merging ###
    original_set_len = len(y)
    x_merge, y_merge, freq_merge = \
        x_unk + x_suggest_unk, \
        y + y_suggest, \
        freq + freq_suggest

    ### discard infrequent labels ###
    # train-validation split
    x_val, x_train, y_val, y_train, freq_val, freq_train, valid_index, statistics_dict = \
        train_validation_split(x=x_merge, y=y_merge, freq=freq_merge, 
                               original_set_len=original_set_len,  
                               label_count_thresh=label_count_thresh, 
                               valid_ratio=valid_ratio, 
                               keep_rare_labels=keep_infreq_labels)

    ### a fork for linear model ###
    # returns a list of dictionaries with counters
    if model == 'linear':
        ### apply ngrams if needed ###
        if mk_ngrams:
            # add filtered ngrams, 
            # introduce UNKNOWN if ngram is not in 'allowed_ngrams'
            unknown_ngram = '<unk-ngram>'
            # for x
            x_train = make_ngrams(x_train, 
                                  allowed_ngrams=allowed_ngrams, 
                                  ngram_width=ngram_width)
#             x_train = [obs + [ngram 
#                               if ngram in allowed_ngrams
#                               else unknown_ngram
#                               for ngram in join_sliding_window(obs, 
#                                   ngram_width)]
#                        for obs in x_train]
            # for suggestion x
            x_val = make_ngrams(x_val, 
                                allowed_ngrams=allowed_ngrams, 
                                ngram_width=ngram_width)
#             x_val = [obs + [ngram 
#                             if ngram in allowed_ngrams
#                             else unknown_ngram
#                             for ngram in join_sliding_window(obs, 
#                                 ngram_width)]
#                        for obs in x_val]
        if linear_counters:
            x_train = [Counter(obs) for obs in x_train]
            x_val = [Counter(obs) for obs in x_val]
        return x_train, x_val, y_train, y_val, allowed_ngrams

    ### a fork for nueral model ###
    # returns an array of indecies and their corresponding dicts
    # i.e. {char:int}, {label:int} and their corresponding inverses
    elif model == 'neural':   
        ### padding ###
        # makes all sequences the same (max) length 
        # pads with 'pad' character
        x_train_filtered_pad, statistics_dict_second = \
            text_filter_pad(text=x_train, y=y_train, 
                            char_filter=char_filter, 
                            filter_keys_chars=allowed_chars)
        # update stats dict
        statistics_dict = {**statistics_dict, 
                           **statistics_dict_second}

        x_val_filtered_pad, _ = \
            text_filter_pad(text=x_val, y=y_val, 
                            char_filter=char_filter, 
                            filter_keys_chars=allowed_chars)

        ### scale up and permute ("shuffle") ###
        # training data
        x_train_scaled, y_train_scaled, _ = \
            scale_permute_data(x=x_train_filtered_pad, 
                               y=y_train, 
                               freq=freq_train, 
                               scale_func=scale_func, 
                               to_permute=to_permute)

        # validation data
        x_val_scaled, y_val_scaled, _ = \
            scale_permute_data(x=x_val_filtered_pad, 
                               y=y_val, 
                               freq=freq_val, 
                               scale_func=scale_func, 
                               to_permute=to_permute)

        ### create look-up dictionaries (and inverse) for an index representation ###
        char_int, char_int_inv, label_int, label_int_inv = \
            lookup_dicts_chars_labels(char_set=statistics_dict['char_set'], 
                                      label_set=statistics_dict['label_set'], 
                                      max_line_len=statistics_dict['seq_len'])

        return x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled,\
            char_int, char_int_inv, label_int, label_int_inv, \
            statistics_dict


def train_validation_split(*, x, y, freq, 
                           original_set_len,
                           label_count_thresh, 
                           valid_ratio, 
                           keep_rare_labels, 
                           seed=seed(),
                           verbose=True):
    assert isinstance(keep_rare_labels, bool), 'note that keep_rare_labels should be a boolean'
    random.seed(seed)
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
    
    label_valid_index_dict = {label:random.sample([ind for ind in indices 
                                                   if ind < original_set_len], 
                              min(
                                int(valid_ratio * len(indices)), 
                                len(
                                    [ind for ind in indices 
                                     if ind < original_set_len]
                                    )
                                  )
                                                  )
                              for label,indices in sorted(label_index_dict.items())}

#     label_valid_index_dict = {label:random.sample(indices, int(valid_ratio * len(indices)))
#                               for label,indices in sorted(label_index_dict.items())}

    # extract all indices into a single set
    valid_index = [item for sublist in list(label_valid_index_dict.values()) for item in sublist]
    valid_index.sort()
    
    # construct 2 sets of variables, validation and training
    allowed_labels = set(y) if keep_rare_labels else set(label_freq_dict.keys())
    if verbose: print('The are {} observations'.format(len(y)))
    x_val, x_train, y_val, y_train, freq_val, freq_train = [], [], [], [], [], []
    for i, (x, y, freq) in enumerate(zip(x, y, freq)):
        if y not in allowed_labels:
            continue
        else:
            if i in valid_index:
                x_val.append(x)
                y_val.append(y)
                freq_val.append(freq)
            else:
                x_train.append(x)
                y_train.append(y)
                freq_train.append(freq)
    if verbose:
        n_val = len(x_val)
        n_train = len(x_train)
        print('Sampling from allowed {} labels'.format(len(allowed_labels)))
        print('{} labels in the validation set, with'.format(len(label_index_dict)))
        print('{} potential observation to draw from.'.format(
            sum([len(v) for v in label_index_dict.values()])))
        print('{} observations sampled for validation'.format(n_val))
        print('{} observations for training'.format(n_train))
        print('The ratio of validation to *training* is about {:.3f}'.format(n_val / n_train))
    
    statistics_dict = {'label_set': allowed_labels, 
                       'n_class': len(allowed_labels)}
    
    return x_val, x_train, y_val, y_train, freq_val, freq_train, valid_index, statistics_dict


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


def scale_permute_data(*, x, y, freq, scale_func, to_permute=True, seed=seed()):
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
        # set seed
        random.seed(seed)
        permute = np.random.permutation(n)
        x_scaled = x_scaled[permute]
        y_scaled = y_scaled[permute]
    
    return x_scaled, y_scaled, n


def dict_addition(input):
    """
    Given a list of dictionaries, 
    the output is a key-wise addition dictionary
    """
    res_dict = {}
    for sub_dict in input:
        res_dict = {key: res_dict.get(key, 0) + sub_dict.get(key, 0)
                    for key in set(res_dict).union(set(sub_dict.keys()))}
    return res_dict


def sliding_window(input_str, width):
    """
    Returns a list with a sliding window
    over the string with given width
    """
    assert len(input_str) >= width, 'Cannot slide with width larger than the string!'
    return [input_str[i:i + width] for i in range(len(input_str) - width + 1)]


def join_sliding_window(input, width):
    """
    Joins a list of strings (by applying a sliding window)
    into a list of contiguously joined strings
    """
    return [''.join(ngram) for ngram 
            in sliding_window(input, width)]


def keep_first_k_chars(*, input, k, 
                       model, 
                       counters=True, 
                       char_int=None, 
                       pad='<pad-char>', 
                       ngram_width=None, 
                       mk_ngrams=None, 
                       allowed_ngrams=None, 
                       unknown_ngram='<unk-ngram>'):
    """
    Take the input and returns a list (or Counters) of the first k elements.
    model should be either 'linear' or 'neural'.
    Use counters to specify if the output should be a list of Counter-s.
    char_int is a dict pointing from characters to their integer encoding.
    If ngrams are given, cutting the sequence short 
    doesn't distinguish between ngrams and characters.
    That's why ngrams needs to be discarded and generated again
    from the shorter sequence.
    
    """
    assert model in ['linear', 'neural'], \
        'Model has to be identified as either linear or neural'
    
    if model == 'linear':
        need_to_make_ngrams = mk_ngrams is True and \
            k >= ngram_width
        # return self or Counters
        func = Counter if counters else lambda x: x
        # return a sliding window, filtered with allowed_ngrams
        filter_join_sliding_window = lambda x, allowed_ngrams: \
            [ngram if ngram in allowed_ngrams else unknown_ngram
             for ngram in join_sliding_window(x, ngram_width)]
        # keep only characters
        keep_chars = lambda x: [elem for elem in x if len(elem)==1]
        # return a slice or a slice and ngrams
        line_rep = lambda x: keep_chars(x)[:k] if not need_to_make_ngrams \
            else keep_chars(x)[:k] + \
            filter_join_sliding_window(keep_chars(x)[:k], allowed_ngrams)
        
        if need_to_make_ngrams:
            assert allowed_ngrams is not None, \
                'If ngrams should be made,' + \
                'please provide the allowed ngrams set.'
        return [func(line_rep(line))
                for line in input]
    
    if model == 'neural':
        pad_enc = char_int[pad]
        return np.asarray([[char_enc if i < k else pad_enc 
                            for i, char_enc in enumerate(line)] 
                           for line in input])


def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_csv(fname, obj, 
             headers=None, 
             encoding='iso-8859-15'):
    """
    Write to a CSV file.
    fname: file name (str).
    obj: the objec to save ([str]).
    headers: a tuple of strings, to be used as headers ([str]).
    """
    with open(fname, 'w', 
              encoding=encoding) as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        if headers is not None:
            writer.writerows([headers])
        writer.writerows(obj)


flatten_list = lambda l: [item for sublist in l for item in sublist]


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

