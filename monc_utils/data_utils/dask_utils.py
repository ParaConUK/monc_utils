# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:33:51 2021

@author: paclk
"""
import numpy as np
import monc_utils
from monc_utils.data_utils.string_utils import get_string_index

from loguru import logger

def re_chunk(f, chunks=None, xch='all', ych='all', zch='auto', tch='auto'):
    """
    Wrapper to re-chunk dask array.
    
    Provides 'all'  as an option to mean chunk = length of dim.

    Parameters
    ----------
    f : dask array.
        Input field
    chunks : dict, optional
        Chunk specification. The default is None.
    xch : str or int, optional
        New chunking for x dimension. 'all' | 'auto' | int .
        The default is 'all'.
    ych : str or int, optional
        New chunking for y dimension. 'all' | 'auto' | int .
        The default is 'all'.
    zch :str or int, optional
        New chunking for z dimension. 'all' | 'auto' | int .
        The default is 'all'.
    tch :str or int, optional
        New chunking for time dimension. 'all' | 'auto' | int .
        The default is 'auto'.

    Returns
    -------
    f : dask array.
        re-chunked input field.

    """

    if monc_utils.global_config['no_dask']:
        return f

    logger.warning('*** Using re_chunk ***')

    defn = 1

    if chunks is None:

        chunks={}
        sh = np.shape(f)
        for ip, dim in enumerate(f.dims):
            if dim == 'x' or dim.startswith('x_'):   
                if xch == 'all':
                    chunks[dim] = sh[ip]
                elif xch == 'auto':
                    chunks[dim] = 'auto'
                else:
                    chunks[dim] = np.min([xch, sh[ip]])
            elif dim == 'y' or dim.startswith('y_'):
                if ych == 'all':
                    chunks[dim] = sh[ip]
                elif ych == 'auto':
                    chunks[dim] = 'auto'
                else:
                    chunks[dim] = np.min([ych, sh[ip]])
            elif dim == 'z' or dim.startswith('z_'):
                if zch == 'all':
                    chunks[dim] = sh[ip]
                elif zch == 'auto':
                    chunks[dim] = 'auto'
                else:
                    chunks[dim] = np.min([zch, sh[ip]])
            elif dim == 'time' or dim.startswith('time_'):
                if tch == 'all':
                    chunks[dim] = sh[ip]
                elif tch == 'auto':
                    chunks[dim] = 'auto'
                else:
                    chunks[dim] = np.min([tch, sh[ip]])
            else:
                chunks[f.dims[ip]] = defn

    f = f.chunk(chunks=chunks)

    return f

def guess_chunk(max_ch, dataset):
    """
    Guess a suitable chunk size for spatial dimensions 

    Parameters
    ----------
    max_ch : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    nch : TYPE
        DESCRIPTION.

    """

    [iix, iiy, iiz] = get_string_index(dataset.dims, ['x', 'y', 'z'])
    xvar = list(dataset.dims)[iix]
    yvar = list(dataset.dims)[iiy]
    zvar = list(dataset.dims)[iiz]

    nch = np.min([int(dataset.dims[xvar]/(2**int(np.log(dataset.dims[xvar]
                                                *dataset.dims[yvar]
                                                *dataset.dims[zvar]
                                                /max_ch)/np.log(2)/2))),
                  dataset.dims[xvar]])
    return nch
