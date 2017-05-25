from itertools import cycle

def chunk_string(iterable, steps=None):
    n = 0
    if steps is None:
        # default lengths of an ATC code
        steps = [1, 2, 1, 1, 2]
    # take care of ATCvet special case
    if iterable[0] == 'Q':
        steps = [1] + steps
    step = cycle(steps)
    while n < len(iterable):
        next_step = next(step)
        yield iterable[n:n + next_step]
        n += next_step
        
chunk_string_list = lambda iterable, steps=None: \
    list(chunk_string(iterable, steps))
    
    
def common_atc_levels(atc_one, atc_two, normalize=False):
    """
    Compare two Atc_code instances, checking how many levels they have in common.
    Returns the number of common levels and a string of the "lowest" common node.
    """
    assert [atc_one.__class__, atc_two.__class__] == \
        [Atc_code, Atc_code], 'Arguments have to be Atc_code instances. Given {}, {}'\
            .format(type(atc_one), type(atc_two))
    levels = 0
    common_node = []
    for a,b in zip(atc_one.atc_parsed, atc_two.atc_parsed):
        if a==b:
            levels += 1
            common_node.append(a)
        else:
            break
    if normalize:
        max_len = max(atc_one.levels, 
                      atc_two.levels)
        levels = float(levels) / max_len
    return levels, ''.join(common_node)
    
    
class Atc_code(object):

    def __init__(self, atc_raw_str):
        self.atc_raw_str = atc_raw_str
        self.atc_parsed = chunk_string_list(self.atc_raw_str)
        self.levels = len(self.atc_parsed)  # num of levels down the tree
        self.is_vet = self.atc_parsed[0] == 'Q'
    
#     def __cmp__(self, other):
#         return self.get_raw() == other.get_raw()
    
    def __repr__(self):
        """
        Define the unambiguous representation of the class instance.
        """
        return 'ATC_code:{}'.format(self.get_raw())
    
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            # this is better comparison for the general case
            # return self.__dict__ == other.__dict__
            
            # but here we can be more concrete and faster
            return self.atc_raw_str == other.atc_raw_str
        return False
    
    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)
        
    def __hash__(self):
        return hash(self.get_raw())
    
    def get_raw(self):
        return str(self.atc_raw_str)
