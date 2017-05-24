
# coding: utf-8

# In[ ]:

from utils.utils import save

from selenium import webdriver

# from itertools import product, islice
from itertools import islice
from time import sleep
from collections import OrderedDict
import re
import argparse


# In[ ]:

"""
Enable passing some keyword arguments from command line.
This does not affect the Jupyter notebook.
"""
# try:
parser = argparse.ArgumentParser()

parser.add_argument('--start', action='store', dest='start',
                    help='Specify the starting iteration number', 
                    type=int,
                    default=None)

parser.add_argument('--stop', action='store', dest='stop',
                    help='Specify the stopping iteration number', 
                    type=int, 
                    default=None)

#     results = parser.parse_args()
args, unknown = parser.parse_known_args()

# if args.logdir is not None and isinstance(args.logdir, str):
#     kwargs_simple_rnn.log_dir = str(args.logdir)

start_itr = args.start
stop_itr = args.stop


# In[ ]:

# all wiki articles of the ATC-code category can be found here
petscan_url = 'https://petscan.wmflabs.org/?psid=1039824'

# some constants regarding the articles in this category
right_table_css_select = '#mw-content-text > table.float-right.infobox.wikitable'
# fields to look for in the table ('Name' is matched from the header)
var_name_list = ['Name', 
                 'ATC-Code', 
                 'Andere Namen', 
                 'CAS-Nummer', 
                 'PubChem', 
                 'Kurzbeschreibung', 
                 'PubMed-Suche', 
                 'Summenformel', 
                 'DrugBank', 
                 'Wirkstoffklasse', 
                 'Wirkmechanismus']


# In[ ]:

driver = webdriver.Firefox()
driver.delete_all_cookies()
driver.get(petscan_url)
driver.implicitly_wait(2)  # seconds


# In[ ]:

# find all table rows that contain the
# given partial_text, on the first (left) table cell
find_row_by_text = lambda partial_text, row_list:     [row for row in row_list 
     if partial_text in row.find_elements_by_tag_name('td')[0].text]

# get the text from the second (right) table cell
get_right_cell_of_row = lambda row:     row.find_elements_by_tag_name('td')[1].text    .split('\n')

strip_ref = lambda text: re.sub(r'\[.+?\]\s*', '', text)

def get_table_data(partial_text, row_list, d=driver):
    """
    Looks for a row in the 'row_list'
    containing 'partial_text' and
    returns (partial_text, [values]).
    In case the partial_text is not found, values are None.
    Removes numbers, used by Wikipedia for references.
    """
    if partial_text == 'Name':
        return (partial_text, 
                [d.find_element_by_id('firstHeading').text])
    atc_row = find_row_by_text(partial_text, right_table_rows)
    if len(atc_row) == 0:
        values = [None]
    else:
        assert len(atc_row) == 1
        atc_row = atc_row[0]
        values = [strip_ref(elem) 
                  for elem in get_right_cell_of_row(atc_row)]
    return (partial_text, values)


# In[ ]:

# get all links tp wiki articles from the ATC-code category 
petscan_results_id = 'main_table'
petscan_results = driver.find_element_by_id(petscan_results_id)
petscan_results = petscan_results.find_elements_by_tag_name('a')
petscan_results_links = [elem.get_attribute('href') 
                         for elem in petscan_results]


# In[ ]:

# iterate through all links and collect results
data_collector_list = []


if start_itr is None:
    start_itr = 1000
if stop_itr is None:
    stop_itr = len(petscan_results_links)
    
print('Starting at {}'.format(start_itr))
print('Stopping at {}'.format(stop_itr))

petscan_results_links = enumerate(petscan_results_links)

iterable = islice(petscan_results_links, 
                  start_itr, 
                  stop_itr)

for step, art_link in iterable:
    print('step number {}'.format(step))
    driver.get(art_link)
    sleep(0.1)
    
    # try, since some pages are really weird, such as
    # https://de.wikipedia.org/wiki/Ginkgo
    try:
        # get rows of the table on the right
        right_table = driver.find_element_by_css_selector(right_table_css_select)
        right_table_rows = right_table.find_elements_by_tag_name('tr')

        # keep rows with exactly 2 columns
        right_table_rows =             [row for row in right_table_rows 
             if len(row.find_elements_by_tag_name('td')) == 2]

        data_collector_list.append(
            OrderedDict([get_table_data(var, right_table_rows) 
                         for var in var_name_list]))
    except:
        print('Encountered a very weird article!')
        print('Step number {}, article named {}'.
              format(step, get_table_data('Name', [])[0]))
        print('Skipping to the next page.')
        


# In[ ]:

print('Done scraping!')
print('Saving now')


# In[ ]:

# print(data_collector_list)

def keep_unique(seq):
    seen = []
    seen_add = seen.append
    return [x for x in seq if not (x in seen or seen_add(x))]

data_collector_list_unique =     keep_unique(data_collector_list)

sub_dir = 'scrape/'

save(fname=sub_dir+'wikipedia_raw_data_collector_num_{}_{}'.     format(start_itr, stop_itr),
     obj=data_collector_list_unique)


# In[ ]:

print('Done saving!')


# In[ ]:

driver.close()


# In[ ]:



