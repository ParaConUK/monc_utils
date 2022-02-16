# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:06:21 2021

@author: paclk
"""
import numpy as np

def bytarr_to_dict(d):

    # Converted for xarray use
    res = {}
    for i in range(np.shape(d)[0]):
        opt = d[i,0].decode('utf-8')
        val = d[i,1].decode('utf-8')

        res[opt] = val
    return res

def options_database(source_dataset):
    '''
    Convert options_database in source_dataset to dictionary.

    Parameters
    ----------
    source_dataset : netCDF4 file
        MONC output file.

    Returns
    -------
    options_database : dict

    '''

    # Converted to xarray

    if 'options_database' in source_dataset.variables:
        options_database = bytarr_to_dict(
            source_dataset['options_database'].values)
    else:
        options_database = None
    return options_database

