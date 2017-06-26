
# coding: utf-8

import tensorflow as tf
# getting data directly from a tensorboard log dir
from tensorflow.python.summary import event_multiplexer

import pandas as pd

from collections import OrderedDict

# from matplotlib import pyplot as plt
# from matplotlib.colors import Normalize
# from numpy import around

import os

import argparse


# In[19]:

"""
Enable passing some keyword arguments from command line.
This does not affect the Jupyter notebook.
"""
# try:
parser = argparse.ArgumentParser()

parser.add_argument('--experiments_dir', action='store', dest='experiments_dir',
                    help='Specify the path to the experiments directory. '+\
                        '(e.g. "/cluster/scratch/raiskiny/experiments_sim_070/")', 
                    type=str,
                    default=None)


# In[20]:

args, unknown = parser.parse_known_args()

if args.experiments_dir is None:
    raise ValueError('Please specify the experiments directory, using the flag "--experiments_dir"')

experiments_dir = args.experiments_dir
# fix file name syntax
if experiments_dir[-1] != '/':
    experiments_dir += '/'


# In[2]:

"""
Scalars tags (not all but most of them):
Mean_Reciprocal_Rank/Mean_Reciprocal_Rank_test
Accuracy/Accuracy_test
In_top_5/In_top_5_test
Cost_function/Total_cost_test
Cost_function_additional_metrics/Cross_entropy_test

attributes
wall_time
step
value
"""


def get_optimal_event(scalars, selection_func):
    """
    Inputs are list of ScalarEvent and a selection function (either min or max).
    e.g. for accuracy, max should be selected, while for cost, min is appropriate.
    Returns the "optimal" event, i.e. the event where the optimal value was achieved.
    """
    if selection_func not in (min, max):
        raise ValueError('selection_func should be either min or max, got unexpected {}'
                         .format(selection_func.__name__))
    
    # optimal value
    optimal_val = selection_func(
        [event.value for event in scalars])
    # optimal event (corresponding to the optimal value)
    optimal_event = [event for event in scalars 
                     if event.value == optimal_val][0]
    
    return optimal_event


def get_all_optimals(log_dir, event_acc):
    """
    Inputs are log_dir (e.g. as from child_dir) 
        and event_acc is the event accumulator.
    Gets optimal events (using 'get_optimal_event') for 
        total cost, Mean Reciprocal Rank and accuracy.
    Returns the three optimal events (optimal_cost, optimal_mrr, optimal_accuracy),
        as well as the three ScalarEvent lists (total_cost, mrr, accuracy)
    """
    # get ScalarEvent lists
    total_cost = event_acc.Scalars(log_dir, 'Cost_function/Total_cost_test')
    mrr = event_acc.Scalars(log_dir, 'Mean_Reciprocal_Rank/Mean_Reciprocal_Rank_test')
    accuracy = event_acc.Scalars(log_dir, 'Accuracy/Accuracy_test')

    # get optimal events
    optimal_cost = get_optimal_event(total_cost, min)
    optimal_mrr = get_optimal_event(mrr, max)
    optimal_accuracy = get_optimal_event(accuracy, max)

    return (optimal_cost, optimal_mrr, optimal_accuracy, 
            total_cost, mrr, accuracy)


def value_at_other_optimal(scalars, optimal_event):
    """
    Inputs are scalars, a list of ScalarEvent and 
        optimal_event, an optimal ScalarEvent.
    Retruns the value from scalars at the optimal_event step.
    e.g. value_at_other_optimal(accuracy, optimal_cost) returns
        the accuracy value at the step where cost was optimal.
    """
    return [event.value for event in scalars 
            if event.step == optimal_event.step][0]


def metrics_dict_from_log(log_dir, 
                          event_acc=None, 
                          optimal_cost=None, 
                          optimal_mrr=None, 
                          optimal_accuracy=None, 
                          total_cost=None, 
                          mrr=None, 
                          accuracy=None):
    """
    Input log_dir is a Tensorboard log directory
        and event_acc is the event accumulator.
    Other inputs are optional and will be generated if not provided.
    Returns an OrderedDict with the model string (log dir name) and evaluation metrics.
    """
    # check if event_acc is None
    if event_acc is None:
        event_acc = event_multiplexer            .EventMultiplexer()            .AddRunsFromDirectory(log_dir)
        event_acc.Reload()
    
    # check if all optional input values are None
    if not any(a is not None 
               for a in [optimal_cost, 
                         optimal_mrr, 
                         optimal_accuracy, 
                         total_cost, 
                         mrr, 
                         accuracy]):
        # get evaluation metrics data from log dir
        (optimal_cost, optimal_mrr, optimal_accuracy, 
         total_cost, mrr, accuracy) = get_all_optimals(log_dir, event_acc)
        
    return OrderedDict(
        [('Model_str', log_dir), 
         ('Cost @ optimal cost', value_at_other_optimal(total_cost, optimal_cost)), 
         ('MRR @ optimal cost', value_at_other_optimal(mrr, optimal_cost)), 
         ('Accuracy @ optimal cost', value_at_other_optimal(accuracy, optimal_cost)), 
         ('step @ optimal cost', optimal_cost.step), 
         
         ('Cost @ optimal MRR', value_at_other_optimal(total_cost, optimal_mrr)), 
         ('MRR @ optimal MRR', value_at_other_optimal(mrr, optimal_mrr)), 
         ('Accuracy @ optimal MRR', value_at_other_optimal(accuracy, optimal_mrr)), 
         ('step @ optimal MRR', optimal_mrr.step), 
         
         ('Cost @ optimal accuracy', value_at_other_optimal(total_cost, optimal_accuracy)), 
         ('MRR @ optimal accuracy', value_at_other_optimal(mrr, optimal_accuracy)), 
         ('Accuracy @ optimal accuracy', value_at_other_optimal(accuracy, optimal_accuracy)),
         ('step @ optimal accuracy', optimal_accuracy.step)
         ])


# In[8]:

# specify path (for parent log dir)
# experiments_dir = './experiments/'

log_parent_dirs = next(os.walk(experiments_dir))[1]

log_parent_dirs = [experiments_dir + elem 
                   for elem in log_parent_dirs]

# log_parent_dirs = [elem for elem in log_parent_dirs
#                    if '4_20' in elem]

# log_parent_dirs = ['logdir_exper_4_20_GRU/',
#                    'logdir_exper_4_20_GRU_bidir/',
#                    'logdir_exper_4_20_LSTM/',
#                    'logdir_exper_4_20_LSTM_bidir/']

# print('cutting short the number of log dirs (else crash)')
# log_parent_dirs = log_parent_dirs[:2]
# log_parent_dirs = log_parent_dirs[:3]


# In[10]:

event_accum = event_multiplexer.EventMultiplexer()
for log_dir in log_parent_dirs:
    event_accum = event_accum.AddRunsFromDirectory(log_dir)

# load
event_accum.Reload()  # this might take a bit, depending on number of runs

# event_accum = {index: event_multiplexer\
#                .EventMultiplexer()\
#                .AddRunsFromDirectory(log_dir)\
#                .Reload()
#                for index, log_dir in enumerate(log_parent_dirs)}


# In[11]:

print('='*50)
print('Done loading {} Tensorboard runs'.format(len(event_accum.Runs())))
print('='*50)


# In[12]:

# get a list of all subfolders in the parent log dir
child_dir = [sub_dir
             for log_dir in log_parent_dirs 
             for sub_dir in next(os.walk(log_dir))[1]]


# In[13]:

parent_dir_metrics = [metrics_dict_from_log(log_dir, event_accum)
                      for log_dir in child_dir]


# In[53]:

exper_metrics = pd.DataFrame(parent_dir_metrics)
# correct the Model_str column to index
# exper_metrics.set_index(keys='Model_str', 
#                         inplace=True, verify_integrity=True)

# sort rows by value
exper_metrics.sort_values(by='MRR @ optimal MRR', ascending=False, 
                          inplace=True)
# exper_metrics.sort_values(by='Cost @ optimal cost', ascending=True, 
#                           inplace=True)

exper_metrics = exper_metrics    .style.background_gradient(
    cmap='spring', low=.5, high=0)\
    .format(  # format all float values
        {col: '{:.2%}' 
         for col in exper_metrics.columns 
         if any(word in col
                for word in ['Accuracy @', 
                             'MRR @', 
                             'Cost @'])})\
    .format(
        {'Cost @ optimal cost' : '{:.3f}', 
             'Cost @ optimal MRR' : '{:.3f}', 
             'Cost @ optimal accuracy' : '{:.3f}'})\
    .apply(lambda x: ["background: greenyellow" # color str cells based on their model type, hacky I know
                      if isinstance(v, str) and 'GRU,bidir=F' in v 
                      else "background: hotpink" if isinstance(v, str) and 'GRU,bidir=T' in v 
                      else "background: coral" if isinstance(v, str) and 'LSTM,bidir=F' in v 
                      else "background: olive" if isinstance(v, str) and 'LSTM,bidir=T' in v 
                      else "" for v in x], 
           axis = 1)
    
df_style = exper_metrics.export()
# can reuse styles with
# Styler.use(exper_metrics.export())


# In[55]:

exper_metrics


# In[50]:

fname = list(filter(None, experiments_dir.split('/')))[-1]
fname += '.csv'

exper_metrics.data.to_csv(
    fname, 
    sep=';')



