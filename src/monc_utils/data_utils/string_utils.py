# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:20:38 2021

@author: Todd Jones
"""

def get_string_index(strings, substrings, exact=False):
    """
    Search for the first index in list of strings that contains substr string
      anywhere or, if exact=True, contains an exact match.

    Parameters
    ----------
        strings      : List of strings to be searched
        substrings   : List of strings to be found in **strings**
        exact        : (bool, optional) Whether to perform search for exact match

    Returns
    --------
        tuple of integers  : first index of match, or None, if not found

    @author: Todd Jones and Peter Clark
    """
    index_list = []

    for substr in substrings:

        idx = None

        for i, string in enumerate(strings):
            if exact:
                if substr == string:
                    idx = i
                    break
            else:
                if substr in string:
                    idx = i
                    break
            # end if (exact)
        index_list.append(idx)

    return tuple(index_list)

def in_list(c:str, strlist:str) -> bool:
    inlist = -1
    for i,item in enumerate(strlist):
        if c in item:
            inlist = True
            break
    return inlist

