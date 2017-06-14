from collections import Counter, OrderedDict
from itertools import combinations
import numpy as np



try:
    from utils.Atc_code import *
except:
    from Atc_code import *



def jaccard_sim(d1, d2):
    # collect union keys
    key_union = list(dict(d1, **d2))
    # for each key, get max(d1[key], d2[key]), with default of zero
    val_union = [max(d1.get(k, 0), d2.get(k, 0)) for k in key_union]
    # collect intersection keys
    key_intersect = [k for k in d1 if k in d2]
    # for each key, get min(d1[key], d2[key]), (with default of inf)
    val_intersect = [min(d1.get(k, inf), d2.get(k, inf)) for k in key_intersect]
    return float(sum(val_intersect) / sum(val_union))


def labeld_unlabeled_similarities(*, sim_thereshold, 
                                  x_unlabeled, X_unlabeled, X_labeled, y):
    """
    Takes in the labeled, unlabeled data and the labels
    Returns sim_result_list, sim_dict_filtered, sim_dict,
    where sim_result_list,
    sim_dict_filtered,
    sim_dict is a dict from unlabeled text to (label, Jaccard sim) tuples for all labeled observations.
    """ 
    sim_dict = {}
    timenow = time()
    for text, features in zip(x_unlabeled, X_unlabeled):
        sim_dict[text] = [(y[i], jaccard_sim(obs, features)) for i, obs in enumerate(X_labeled)]

    print('Calculating all Jaccard similarities took {:.2f} seconds'.format(time() - timenow))
    
    # filtering out entries with similarity below threshold
    timenow = time()
    sim_dict_filtered = {}
    for k, v in sim_dict.items():
        sim_dict_filtered[k] = [(label, sim) for (label, sim) in v if sim >= sim_thereshold]
        # remove labels that don't meet the criteria
        if sim_dict_filtered[k] == []: del sim_dict_filtered[k]    
    print('Removing entries with Jaccard sim lower than the threshold took {:.2f} seconds'.
          format(time() - timenow))
    
    # return only the top result
    sim_result_list = []
    for k, v in sim_dict_filtered.items():
        for label, sim in v:
            # keep the entry with the highest similarity
            top_sim = 0
            if sim > top_sim:
                top_label, top_sim = label, sim
            sim_result_list.append([k, top_label, top_sim])
    
    return sim_result_list, sim_dict_filtered, sim_dict


def get_filtered_suggestion_results(threshold, 
                                    sim_dict_filtered):
    """
    Inputs are threshold, a Jaccard similarity threshold
        and sim_dict_filtered, of the form {'text': [(label, similarity)]}.
    Returns some summaries regarding the label suggestions.
        text: the free text field.
        labels: the suggested labels.
        number_of_distinct_labels: self explaining.
        labels_sim_tup: retain the tuples in the raw form.
    """
    text = []
    labels = []
    number_of_distinct_labels = []
    labels_sim_tup = []

    for key,val in sim_dict_filtered.items():
        # val is a list of (label, sim) tuples

        if max({v[1] for v in val}) >= threshold:
            
            # add the free text
            text.append([key])
            # add the labels
            labels.append([v[0] for v in val 
                           if v[1] >= threshold])
            # count the number of distinct labels
            number_of_distinct_labels.append(
                len({v[0] for v in val 
                     if v[1] >= threshold}))
            # add the filtered tuples list
            labels_sim_tup.append([v for v in val 
                                   if v[1] >= threshold])
            
    return text, labels, \
           number_of_distinct_labels, \
           labels_sim_tup


def combinations_mod(iterable, r):
    """
    Modifying itertools.combinations, so it can be used on a single object (return the same onject twice).
    """
    if len(iterable) == 1:
        x = iterable.pop()
        return combinations([x, x], r)
    else:
        return combinations(iterable, r)
        

def record_threshold_data(threshold, 
                          number_of_distinct_labels, 
                          labels_sim_tup, 
                          x_unlabeled, 
                          y, 
                          kwargs_lin_data_init):
        """
        A handling function to record summaries for each threshold.
        Is used to generate a Pandas DataFrame to analyze suggestions.
        """
        
        # get list of atc codes, for each suggestion
        atc_combinations = [
            {atc for atc, sim in tup} 
            for tup in labels_sim_tup]

        # convert the atc code lists to combinations
        atc_combinations = [combinations_mod(combo, 2) 
                            for combo in atc_combinations]

        # calculate the average common ATC levels, per suggestion
        # subscript [0] to get the number of levels ([1] gives the ATC code string)
        atc_common = \
            [np.mean([common_atc_levels(*pair, normalize=True)[0]
              for pair in combo])
             for combo in atc_combinations]
        
        # Mean common ATC levels, non-unanimous (norm)
        # handle empty lists
        non_unanimous_levels = \
            [level 
             for level in atc_common 
             if level != 1.0]
        if non_unanimous_levels == []:
            non_unanimous_levels = None
        else:
            non_unanimous_levels = \
                np.mean(non_unanimous_levels)
                
        # count unanimous suggestions
        suggestions = [
            {atc 
             for atc,sim in suggestion_lists} 
            for suggestion_lists in labels_sim_tup]

        unanimous_suggestions = [suggest.pop().get_raw() 
                                 for suggest in suggestions 
                                 if len(suggest) == 1]
        
        # count how many labels will be left after filtering out infrequent labels
        labels_counter_filter = \
            {k:v for k,v 
             in Counter(unanimous_suggestions + y).items() 
             if v >= kwargs_lin_data_init.label_count_thresh}
        
        row = OrderedDict()
        row['Threshold'] = threshold
        row['Mean # suggestions'] = np.mean(number_of_distinct_labels)
        row['Max # suggestions'] = max(number_of_distinct_labels)
        row['Mean similarity'] = \
            np.mean([[sim 
                      for tup in labels_sim_tup 
                      for atc,sim in tup]])
        row['Min similarity'] = \
            min([sim 
                 for tup in labels_sim_tup 
                 for atc,sim in tup])
        row['Unanimous suggestions'] = \
            len(unanimous_suggestions)
        row['No unanimous suggestion'] = \
            len(x_unlabeled) - row['Unanimous suggestions']
        # max similarity, among all similarities, pointing to the same ATC code
        row['Unanimous mean similarity'] = \
            np.mean([max([tup[it][1] for it in range(len(tup))])
                     for tup, l 
                     in zip(labels_sim_tup, 
                            number_of_distinct_labels) 
                     if l == 1])
        row['Mean common ATC levels (norm)'] = \
            np.mean(atc_common)
        row['Mean common ATC levels, non-unanimous (norm)'] = \
            non_unanimous_levels
        row['Unanimous suggestions and labeled data, filtered (infrequent labels)'] = \
            sum(labels_counter_filter.values())
        return row
    
    
# sample "sample_count" times from "input_list"
sample_from_list = lambda input_list, sample_count: \
    [input_list[i] 
     for i in 
     sorted(random.sample(range(len(input_list)), 
                          sample_count))]



if __name__ == '__main__':
    pass
