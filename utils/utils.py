import getpass
# getpass.getuser()
# 'yarden'

import pickle
#import re
import inspect

def seed():
    return 2178


def user_opt_gen():
    user_opt = {
        'yarden' : {
            'data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/20170303_EXPORT_for_Yarden.csv',
    #         'data_path' : r'/home/yarden/git/Automated_categorization_medication/data/20170303_EXPORT_for_Yarden.csv',
            'atc_conversion_data_path' : r'/media/yarden/OS/Users/Yarden-/Desktop/ETH Autumn 2016/Master Thesis/Data/Complete_ATCs_and_lacking_translations_V03a_20161206.csv'
        },
        'Yarden-' : {
            'data_path' : None, 
            'atc_conversion_data_path' : None
        }
    }
    cur_user = getpass.getuser()
    return user_opt[cur_user]


def save(fname, obj):
    with open(fname, 'w') as f:
        pickle.dump(obj, f)


def load(fname):
    with open(fname, 'r') as f:
        return pickle.load(f)


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

