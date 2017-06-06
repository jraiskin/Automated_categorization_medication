from utils.utils import nice_dict, unscale, seed
from numpy import inf

# from utils.kwargs_file import kwargs_neural_data_init
# from utils.kwargs_file import kwargs_lin_clf, kwargs_svm

# from utils.kwargs_file import kwargs_lin_clf, kwargs_svm, \
#     kwargs_rnn, kwargs_neural_data_init


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

kwargs_rnn = nice_dict({
    # log
    'log_dir': 'logdir/', 
    'del_log': True, 
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
    'hidden_state_size': 32, 
    'keep_prob': 0.7, 
    # noisy activation hyper params
    'activation_function': 'tf.tanh',  # tf.tanh / 'noisy_tanh'
    'learn_p_delta_scale': False,   # noise scale param in noisy activation
    'noise_act_alpha': 1.15,  # mixing in the linear activation
    'noise_act_half_normal': False,
    # regularization constants
    'l2_weight_reg': 1E-3, 
    'target_rep': True, 
    'target_rep_weight': 0.3, 
    # training settings
    'epochs': 1000,
    'summary_step': 10, 
    'save_step': inf,
    'to_save': False, 
    'verbose_summary': False
})

# SVM model kwargs
kwargs_lin_data_init = nice_dict({'model': 'linear', 
                                  'mk_chars': mk_chars, 
                                  'char_filter': char_filter, 'allowed_chars': None, 
                                  'mk_ngrams': mk_ngrams_lin, 'ngram_width': ngram_width, 
                                  'ngram_filter': ngram_filter, 'allowed_ngrams': None, 
                                  'keep_infreq_labels': keep_infreq_labels, 'label_count_thresh': label_count_thresh, 
                                  'valid_ratio': valid_ratio, 
                                  'scale_func': unscale, 'to_permute': True, 
                                  'use_suggestions': use_suggestions
                                  })

kwargs_svm = nice_dict({'C': 0.10,  # penalty term
                        'decision_function_shape': 'ovr',  # one-vs-rest (‘ovr’) / one-vs-one (‘ovo’) 
                        'random_state': seed(), 
                        'kernel': 'linear', 
                        'gamma': 'auto' ,  # kernel coef for ‘rbf’, ‘poly’ and ‘sigmoid’. ‘auto’ -> 1/n_features
                        'probability': True,  # enable probability estimates 
                        'shrinking': True,  # use the shrinking heuristic 
                        'max_iter': -1  # -1 mean no limitation 
                        })

# logistic regression kwargs
kwargs_logistic = nice_dict({'C': 1.0, 
                             'penalty': 'l2', 
                             'multi_class': 'ovr', # one-vs-rest (‘ovr’) / one-vs-one (‘ovo’) 
                             'random_state': seed(), 
                             'solver': 'newton-cg',  # ‘liblinear’ is fit to "small" data-sets, crashes kernel 
                             # solver:{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}, default: ‘liblinear’
                             'fit_intercept': True
                             })

