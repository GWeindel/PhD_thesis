# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:51:35 2018

@author: Laure Spieser and Boris Burle
Laboratoire de Neurosciences Cognitives
UMR 7291, CNRS, Aix-Marseille Université
3, Place Victor Hugo
13331 Marseille cedex 3
"""


def is_int(val):
    """Tests whether val is integer
    """
    try : 
        val = float(val)
    except:
        return False
    else:        
        r = int(val) - float(val)
        if r == 0 : return True
        else: return False

def try_int(val):
    """Returns int(val) if val is integer, returns val otherwise.
    """
    if is_int(val):
        return int(float(val))
    else: return val

def is_num(val):
    """Returns float(val) if val is numeric, returns val otherwise.
    """
    try : val = float(val)
    except : 
        return False
    return True

def try_num(val):
    """Returns int(val)/float(val) if val is integer/float, returns val otherwise.
    """
    if is_int(val):
        return int(float(val))
    elif is_num(val):
        return float(val)
    else: return val
    
def remove_list(list_of_list, empty=None, keep_index=0):
    """Removes lists contained in another list.
    """    
    if isinstance(list_of_list, str) :
        list_of_list = str_to_list(list_of_list)
    # if we have a lists from inside a list
    if hasattr(list_of_list,'__iter__') and not isinstance(list_of_list, str):
        new_list = []
        for li in list_of_list:
            if isinstance(li, str):
                li = str_to_list(li)
            while hasattr(li,'__iter__') and not isinstance(li,str):
                if len(li) > 0 : li = li[keep_index]
                else: li = empty
            # check for empty strings
            if isinstance(li,str) and not li:
                li = empty
            new_list.append(li)                
        if len(new_list) == 0:
            new_list = empty
        elif len(new_list) == 1:
            new_list = new_list[0]
        return new_list    
    else: 
        return list_of_list
    
def in_list(anything):
    """Return anything in a list.
    """    
    if hasattr(anything,'__iter__'):
        if not isinstance(anything, str):
            return list(anything)
        elif len(anything) == 0: 
            return []        
        else: 
            return [anything]
    else: 
        return [anything]

def in_array(anything):
    """Return anything in a numpy array.
    """    
    import numpy as np
    if hasattr(anything,'__iter__'):
        if not isinstance(anything, str):
            return np.array(anything)
        elif len(anything) == 0: 
            return np.array([])        
        else: 
            return np.array(anything)
    else: 
        return np.array(anything)

def get_int_list(code, ignore_starting=[], ignore_space=True):
    """Return a list with only integers, extracted from list of strings.
        e.g., ['S 1', 'S 2', 'S 1'] returns [1,2,1].
    """    
    int_idx = []
    int_val = []
    for idx,c in enumerate(code):
        if ignore_space is True:
            c = c.replace(' ','')
        for i in in_list(ignore_starting):
            if c.find(i) == 0:
                c = c.replace(i,'')
        if is_int(c):
            int_idx.append(idx)
            int_val.append(int(c))
    return (int_idx,int_val)

def str_to_list(any_string, opener='[', closer=']'):
    """Returns a list extracting from a string.
    """    
    if (len(any_string) > 0) and (any_string[0] == opener) and (any_string[-1] == closer):
        return [try_num(_) for _ in any_string[1:-1].split(',')]
    else:
        return(any_string)

#def getIntValues(code, ignore_starting=[], ignore_space=True):
#    int_idx = getIntIdx(code, ignore_starting=ignore_starting, ignore_space=ignore_space)
#    for idx in int_idx:
#        if ignore_space is True:
#            c = c.replace(' ','')
#        for i in utils.inList(ignore_starting):
#            if c.find(i) == 0:
#                c = c.replace(i,'')
#                break
        