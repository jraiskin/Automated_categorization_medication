
# coding: utf-8

# In[ ]:

from utils.utils import save

from selenium import webdriver

from itertools import product, islice
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

url = 'https://compendium.ch/identa/searchPills.aspx?Platform=Desktop'


# In[ ]:

driver = webdriver.Firefox()
driver.delete_all_cookies()
driver.get(url)


# In[ ]:

# driver.find_elements_by_xpath('//*[@id]')
drop_down_menus_xpath = ['//*[@id="ctl00_MainContent_ddlGalForm"]',
 '//*[@id="ctl00_MainContent_ddlGeoForm"]',
 '//*[@id="ctl00_MainContent_ddlColor"]',
 '//*[@id="ctl00_MainContent_ddlNotch"]', 
 '//*[@id="ctl00_MainContent_ddlLength"]']

click = lambda item_id, d=driver: d.find_element_by_id(item_id).click()
search_id = 'ctl00_MainContent_btnSearch'
reset_id = 'ctl00_MainContent_btnReset'
# res_table_id = 'ctl00_MainContent_gvwPills'
# res_table_id = 'ctl00_MainContent_gvwPills_ctl02_lblDescr'  # first row, text cell?
product_list_left_id = 'listProducts'
product_details_ids = ['ctl00_MainContent_ucProductDetail1_dvProduct_lblProductDescr', 
                       'ctl00_MainContent_ucProductDetail1_dvProduct_lblAtcDescr', 
                       'ctl00_MainContent_ucProductDetail1_dvProduct_lblAtcCode', 
                       'ctl00_MainContent_ucProductDetail1_dvProduct_lblKCH']

# check if no results returned
no_results = lambda:     len(driver.find_elements_by_id('ctl00_MainContent_lblNoDataFound')) != 0
# map from drop-down menu options to the correct range,
# to be used in the option's xpath
# special case for length menu:
# start range from 1, to check if there are no results (then skip)
drop_down_index = lambda x, num: list(range(1, len(x) + 2))     if len(x) == num else     list(range(2, len(x) + 2))

find_digits = lambda s: [int(num) for num in re.findall('\d+', s)]

top_left_panel_digits = lambda:     find_digits(
        driver.find_element_by_id(
            'ctl00_MainTitle_lblTitle').text)

too_many_res = lambda:     len(top_left_panel_digits()) > 0 and     top_left_panel_digits()[0] >= 100
    
click_results_table = lambda d=driver:     d.find_element_by_xpath('//*[@id="ctl00_MainContent_gvwPills"]/tbody/tr[1]/td[2]')    .click()


# In[ ]:

def log_results(d=driver, 
                product_list_left_id=product_list_left_id, 
                product_details_ids=product_details_ids):
    """
    Returns a list of OrederedDict with the data we wish to collect.
    """
    
    res_list = []

    for num in range(top_left_panel_digits()[1]):
        # get product links (updates after each click / fetch)
        product_list_left = d.find_element_by_id(product_list_left_id)
        product_links = product_list_left.find_elements_by_tag_name('a')
        if 'num_of_products' not in globals():
            num_of_products =  len(product_links)
            assert num_of_products == top_left_panel_digits()[1]
        
        product_links[num].click()
        
        sleep(2.5)
        
        res_list.append(
            OrderedDict(
                [(item_id, driver.find_element_by_id(item_id).text) 
                 for item_id in product_details_ids]))
    return res_list


# In[ ]:

# populate all dropdown menu options
drop_down_menu_vals = {}
for xpath in drop_down_menus_xpath:
    elem = driver.find_elements_by_xpath(xpath)
    assert len(elem) == 1, 'Item shuold be unique.'
    elem = elem[0]
    values = [i.text # i.get_attribute('value') 
              for i in elem.find_elements_by_tag_name('option') 
              if i.text != 'Alle']
    drop_down_menu_vals[
        elem.get_attribute('id')] = values
    
# number of elements in the "length" menu
length_menu_num =     len(drop_down_menu_vals['ctl00_MainContent_ddlLength'])


# In[ ]:

# get the id's from the xpaths
menu_ids = [elem.partition('"')[-1].rpartition('"')[0] 
            for elem in drop_down_menus_xpath]
# make sure
assert set(menu_ids) == set(drop_down_menu_vals.keys())


# In[ ]:

driver.get(url)
click(reset_id)


# In[ ]:

menu_combinations =     [drop_down_index(drop_down_menu_vals[key], 
                     length_menu_num)
     for key in menu_ids]

# create all Cartesian products
menu_combinations = product(*menu_combinations)
# enumerate for logging purposes
menu_combinations = enumerate(menu_combinations)


# In[ ]:

# an emaple of how all of the dropdown menus combinations might be clicked on
driver.implicitly_wait(2)  # seconds
# Set the amount of time to wait for a page load to complete before throwing an error
# driver.set_page_load_timeout(2.0)
# Set the amount of time that the script should wait during an
# execute_async_script call before throwing an error
# driver.set_script_timeout(2.0)

data_collector_list = []

if start_itr is None:
    start_itr = 392
if stop_itr is None:
    stop_itr = 500

iterable = islice(menu_combinations, 
                  start_itr, 
                  stop_itr)
for itr in iterable:
    click(reset_id)    
    sleep(0.5)
    
    step, cart_prod = itr
    # set the drop-down menu state
    for j in range(len(cart_prod)):
        xpath = drop_down_menus_xpath[j]+'/option[{}]'.            format(cart_prod[j])
        driver.find_element_by_xpath(xpath).click()
            
    print('step number {}'.format(step))
#     print('product is {}'.format(list(zip(menu_ids, 
#                                      cart_prod))))

    # perform search
    click(search_id)
    sleep(0.2)
    
    if cart_prod[j] == 1:  # if lengths = "Alle" 
        if no_results():  # returns no results
            # it is safe to skip all iterations on lengths menu
            try:
                [iterable.__next__() 
                 for _ in range(length_menu_num)]
            except:
                # if we've exceeded the iterator's length
                # it's still ok, since the next "islice batch"
                # would not lose any iteration with information
                pass
        elif too_many_res():  # returns too many results
            continue  # iterate over lengths in the length menu
        else:
            click_results_table()
            sleep(2.2)  # wait for new page to load
            data_collector_list =                 data_collector_list + log_results()
            driver.get(url)
            sleep(0.2)
            # it is safe to skip all iterations on lengths menu
            try:
                [iterable.__next__() 
                 for _ in range(length_menu_num)]
            except:
                # if we've exceeded the iterator's length
                # it's still ok, since the next "islice batch"
                # would not lose any iteration with information
                pass
#     elif no_results():
#         pass
    else:
        if no_results():
            continue
        else:
            click_results_table()
            sleep(2.2)  # wait for new page to load
            data_collector_list =                 data_collector_list + log_results()
            driver.get(url)
            sleep(0.2)


# In[ ]:

print('Done scraping!')
print('Saving now')


# In[ ]:

# print(data_collector_list)

def keep_unique(seq):
    seen = []
    seen_add = seen.append
    return [x for x in seq if not (x in seen or seen_add(x))]

# keep_unique(data_collector_list)

sub_dir = 'scrape/'

save(fname=sub_dir+'data_collector_list_num_{}_{}'.format(start_itr, stop_itr), 
     obj=data_collector_list)

save(fname=sub_dir+'data_collector_list_unique_num_{}_{}'.format(start_itr, stop_itr), 
     obj=keep_unique(data_collector_list))


# In[ ]:

print('Done saving!')


# In[ ]:

driver.close()


# In[ ]:



