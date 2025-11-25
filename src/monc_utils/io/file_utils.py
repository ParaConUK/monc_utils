"""
Created on Mon Aug  2 11:06:21 2021.

@author: Peter Clark and Todd Jones
"""
import numpy as np
import re


def _bytarr_to_dict(d):

    # Converted for xarray use

    while len(np.shape(d))>2:
        d = d[0]
    res = {}
    for i in range(np.shape(d)[0]):
        opt = d[i,0].decode('utf-8')
        val = d[i,1].decode('utf-8')

        res[opt] = val
    return res

def options_database(source_dataset):
    """
    Convert options_database in source_dataset to dictionary.

    Parameters
    ----------
    source_dataset : netCDF4 file
        MONC output file.

    Returns
    -------
    options_database : dict

    """
    # Converted to xarray

    if 'options_database' in source_dataset.variables:
        options_database = _bytarr_to_dict(
            source_dataset['options_database'].values)
    else:
        options_database = None
    return options_database

def path_to_resolution(inpath):
    """
    Pull resolution value from an encoded path as a float.

    e.g., 'BOMEX_m0020_g0800'
    i.e., it should have '_m[0-9][0-9][0-9]' (at least 3 integers)

    Parameters
    ----------
    inpath : str
        file path

    Returns
    -------
    dx : float
        MONC horizontal resolution [m]
    """
    fullpath = inpath
    # Allow for variable length integer string.
    usc = [i for i, ltr in enumerate(fullpath) if ltr not in '0123456789']
    usc.append(len(fullpath))
    usc = np.asarray(usc)
    mnc = [m.start(0) for m in re.finditer('_m[0-9][0-9][0-9]', fullpath)][-1] + 2
    enc = usc[usc > mnc].min()
    dx = float(fullpath[mnc:enc])

    return dx

def configure_model_resolution(dataset, options=None):
    """
    Find model resolution from available sources.

    This routine applies an order of precedence between potential
    pre-existing records of MONC horizontal resolutions and ensures
    that the options dictionary contains these values.

    Files written via io/dataout.py's setup_child_file will have
    the correct value listed in the file's global attributes, as this
    routine is called within that space.

    Repeated calls to this routine (for instance, to simply obtain
    dx and dy) will not modify the options contents.

    Precedence:
       1. options_database
       2. dataset attributes
       3. monc_utils options
       4. parse file path containing resolution encoding

    Parameters
    ----------
    dataset : xarray dataset
        any monc_utils-compatible dataset
    options : dict
        dataset-associated options dictionary

    Returns
    -------
    dx : float (expected)
        x-direction MONC resolution [m]
    dy : float (expected)
        y-direction MONC resolution [m]
    options : dict
        input options dictionary, possibly updated with dx and dy keys
    """
    od = options_database(dataset)
    attrs = dataset.attrs

    # 1st priority: pull from options_database, if present
    if type(od) is dict:
        dx = float(od['dxx'])
        dy = float(od['dyy'])
    # 2nd priority: pull from dataset attributes
    elif ('dx' in attrs and 'dy' in attrs):
        dx = attrs['dx']
        dy = attrs['dy']
    # 3rd priority: use values present in options
    elif options is not None and 'dx' in options and 'dy' in options:
        dx = options['dx']
        dy = options['dy']
    # 4th priority: parse file path for coded info
    elif options is not None and 'input_file' in options:
        dx = path_to_resolution(options['input_file'])
        dy = dx
    else:
        raise RuntimeError("Cannot determine grid resolution.")

    if options is not None:
        options['dx'] = dx
        options['dy'] = dy

    return dx, dy, options
