# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:20:38 2021

@author: Todd Jones
"""

def get_string_index(strings, substrings, exact=False):
    """
    Searches for the first index in list of strings that contains substr string
      anywhere or, if exact=True, contains an exact match

    Args:
        strings      : List of strings to be searched
        substrings   : List of strings to be found in **strings**
        exact        : (bool, optional) Whether to perform search for exact match

    Returns:
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
