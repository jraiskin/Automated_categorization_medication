import pandas as pd
try:
    from utils.utils import user_opt_gen, nice_dict, seed
except:
    from utils import user_opt_gen, nice_dict, seed
    

# create a charachter:count dict
def char_freq_map(*, input_data, filter_by_chars = None, **kwargs):
    char_dict = {}
    unknown = '<unk-char>'
    # check if dataframe or a single obs
    if isinstance(input_data, pd.core.series.Series):
        # getting line
        for line in input_data:
            # splitting into characters
            chars = list(line)
            for char in chars:
                if filter_by_chars == None or char in filter_by_chars:
                    char_dict[char] = char_dict.get(char, 0) + 1
                else:
                    char_dict[unknown] = char_dict.get(unknown, 0) + 1
    elif isinstance(input_data, str):
        # splitting into characters
            chars = list(input_data)
            for char in chars:
                if filter_by_chars == None or char in filter_by_chars:
                    char_dict[char] = char_dict.get(char, 0) + 1
                else:
                    char_dict[unknown] = char_dict.get(unknown, 0) + 1
    return nice_dict(char_dict)


# create a ngram:count dict
def ngram_freq_map(*, input_data, width, filter_by_keys = None, **kwargs):
    ngram_dict = {}
    # check if dataframe or a single obs
    if isinstance(input_data, pd.core.series.Series):
        # getting line
        for line in input_data:
            ngram_dict = update_ngram_dict(line, width, ngram_dict, filter_by_keys)
    elif isinstance(input_data, str):
        ngram_dict = update_ngram_dict(input_data, width, ngram_dict, filter_by_keys)
    return nice_dict(ngram_dict)


# create a sliding window and update a dict with counts (default 0)
def update_ngram_dict(line, width, ngram_dict, filter_by_keys, **kwargs):
    ngrams = sliding_window(line, width)
    unknown = '<unk-ngram>'
    for ngram in ngrams:
        if filter_by_keys == None or ngram in filter_by_keys:
            ngram_dict[ngram] = ngram_dict.get(ngram, 0) + 1
        else:
            ngram_dict[unknown] = ngram_dict.get(unknown, 0) + 1
    return ngram_dict


def filter_dict_by_val_atleast(input_dict, value):
    return nice_dict({k:input_dict[k] for k in input_dict if input_dict[k] >= value})


# returns a list with a sliding window
# over the string with given width
def sliding_window(input_str, width):
    assert len(input_str) >= width, 'Cannot slide with width larger than the string!'
    return [input_str[i:i + width] for i in range(len(input_str) - width + 1)]


# create a joint dict for every observation in the input_data
# based on 'ngram_freq_map' and 'char_freq_map'
# enables to only select one feature type and filtering
def lin_clf_features(*, input_data,
                     mk_ngrams=None, width, ngram_filter, filter_keys_ngrams=None, 
                     mk_chars=None, char_filter, filter_keys_chars=None, 
                     **kwargs):
    assert (mk_ngrams or mk_chars), 'Please select either to create n-grams or character features.'

    if mk_ngrams:
        # filter ngrams to only those that appear at least 'ngram_filter' times in the input
        if isinstance(ngram_filter, int):
            print('N-grams filter is applied.')
            if filter_keys_ngrams is None:
                filter_keys_ngrams = list(
                    filter_dict_by_val_atleast(
                        input_dict=ngram_freq_map(input_data=input_data, 
                                                  width=width), 
                        value=ngram_filter)
                    .keys())
            else:
                print('N-grams filter has been detected, using those keys as filters.')
                filter_keys_ngrams = filter_keys_ngrams
            # apply ngram_freq_map, after figuring out which keys to keep
            X_features_ngrams = [ngram_freq_map(input_data=obs, 
                                                width=width, 
                                                filter_by_keys=filter_keys_ngrams) for obs in input_data]
        # if no filter, just apply the function (for all keys)
        else:
            print('N-grams filter is NOT applied')
            X_features_ngrams = [ngram_freq_map(input_data=obs, 
                                                width=width)
                     for obs in input_data]
    else:
        X_features_ngrams = [{} for ind in range(len(input_data))]
        
    if mk_chars:
        # filter by character, appear at least 'char_filter' times in the input
        if isinstance(char_filter, int):
            print('Character filter is applied')
            if filter_keys_chars is None:
                filter_keys_chars = list(
                    filter_dict_by_val_atleast(
                        input_dict=char_freq_map(input_data=input_data), 
                        value=char_filter)
                    .keys())
            else:
                print('Character filter has been detected, using those keys as filters.')
                filter_keys_chars = filter_keys_chars
            # apply ngram_freq_map, after figuring out which keys to keep
            X_features_chars = [char_freq_map(input_data = obs, 
                                              filter_by_chars=filter_keys_chars) 
                                for obs in input_data]
        else:
            print('Character filter is NOT applied')
            X_features_chars = [char_freq_map(input_data = obs) 
                                for obs in input_data]
    else:
        X_features_chars = [{} for ind in range(len(input_data))]
    
    # merge two dicts, also return the filter n-grams and characters keys.
    return [nice_dict({** X_features_ngrams[ind] ,**X_features_chars[ind]}) 
              for ind in range(len(input_data))], filter_keys_ngrams, filter_keys_chars


if __name__ == '__main__':
    pass

