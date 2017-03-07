import getpass
# getpass.getuser()
# 'yarden'

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

def user_opt_gen():
    cur_user = getpass.getuser()
    return user_opt[cur_user]

#if __name__ == '__main__':
#    cur_user = getpass.getuser()
#    user_opt = user_opt[cur_user]
