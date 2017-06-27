from utils.utils import nice_dict, unscale, seed
from numpy import inf

#### setting hyper-parameters ####
# characters
mk_chars = True
char_filter = 2

# ngrams
mk_ngrams_rnn = False
mk_ngrams_lin = True
ngram_width = 5
ngram_filter = 2

# other pre-processing parameters
use_suggestions = '0.8'
keep_infreq_labels = False
label_count_thresh = 10
valid_ratio = 0.25

# RNN model kwargs
kwargs_neural_data_init = nice_dict(
    {'model': 'neural', 
     'mk_chars': mk_chars, 
     'model': 'neural', 
     'char_filter': char_filter, 'allowed_chars': None, 
     'mk_ngrams': mk_ngrams_rnn, 'ngram_width': ngram_width, 
     'ngram_filter': ngram_filter, 'allowed_ngrams': None, 
     'keep_infreq_labels': keep_infreq_labels, 'label_count_thresh': label_count_thresh, 
     'valid_ratio': valid_ratio, 
     'scale_func': unscale, 'to_permute': True, 
     'use_suggestions': use_suggestions})

kwargs_rnn_GRU = nice_dict({
    # log
    'log_dir': 'rnn_final_batch/sim_10/', 
    'del_log': False, 
    # preprocessing and data
    'scale_func': kwargs_neural_data_init.scale_func, 
    'keep_infreq_labels': kwargs_neural_data_init.keep_infreq_labels, 
    'top_k': 5, 
    'seed': seed(), 
    # learning hyper-params
    'learn_rate': 1E-2, 
    'dynamic_learn_rate': False, 
    'rnn_type': 'GRU',
    'bidirection': False, 
    'char_embed_dim': 4, 
    'one_hot': True,
    'hidden_state_size': 128, 
    'keep_prob': 1.0, 
    # noisy activation hyper params
    'activation_function': 'noisy_tanh',  # tf.tanh / 'noisy_tanh'
    'learn_p_delta_scale': False,   # noise scale param in noisy activation
    'noise_act_alpha': 0.9,  # mixing in the linear activation
    'noise_act_half_normal': True,
    # regularization constants
    'l2_weight_reg': 1E-2, 
    'target_rep': True, 
    'target_rep_weight': 0.3, 
    # training settings
    'epochs': 3000,
    'summary_step': 10, 
    'save_step': 400,
    'to_save': True, 
    'verbose_summary': False
})

kwargs_rnn_LSTM = nice_dict({
    # log
    'log_dir': 'rnn_final_batch/sim_10/', 
    'del_log': False, 
    # preprocessing and data
    'scale_func': kwargs_neural_data_init.scale_func, 
    'keep_infreq_labels': kwargs_neural_data_init.keep_infreq_labels, 
    'top_k': 5, 
    'seed': seed(), 
    # learning hyper-params
    'learn_rate': 1E-2, 
    'dynamic_learn_rate': False, 
    'rnn_type': 'LSTM',
    'bidirection': False, 
    'char_embed_dim': 4, 
    'one_hot': True,
    'hidden_state_size': 128, 
    'keep_prob': 0.7, 
    # noisy activation hyper params
    'activation_function': 'noisy_tanh',  # tf.tanh / 'noisy_tanh'
    'learn_p_delta_scale': True,   # noise scale param in noisy activation
    'noise_act_alpha': 1.15,  # mixing in the linear activation
    'noise_act_half_normal': True,
    # regularization constants
    'l2_weight_reg': 1E-3, 
    'target_rep': True, 
    'target_rep_weight': 0.5, 
    # training settings
    'epochs': 3000,
    'summary_step': 10, 
    'save_step': 400,
    'to_save': True, 
    'verbose_summary': False
})

