"""
Created on Mon Aug  2 11:29:37 2021

@author: Peter Clark
"""
import numpy as np
import monc_utils
import xarray as xr

import monc_utils.data_utils.difference_ops as do

from monc_utils.io.datain import get_data
from monc_utils.io.dataout import save_field
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.data_utils.dask_utils import re_chunk

from loguru import logger

def deformation(source_dataset, ref_dataset, derived_dataset,
                options, grid='w', uvw_names=["u","v","w"]) :
    r"""
    Compute deformation tensor.

    Deformation tensor is defined as :math:`{\partial u_{i}}/{\partial {x_j}}`.

    Parameters
    ----------
        source_dataset  : NetCDF dataset
            Inout data.
        ref_dataset     : NetCDF dataset
            Input data for input containing reference
            profiles. Can be None
        derived_dataset : NetCDF dataset
            Output dataset for derived data.
        options         : dict
            General options e.g. FFT method used.
        grid : str
            destination grid (Default = 'w')
        uvw_names: list of str
            specific names for u, v, and w fields when differing from MONC default
            Required to be in this order, otherwise we can change to 3 parameters?

    Returns
    -------
        xarray
            Array with new dimensions 'i' and 'j'.
            Saves to derived_dataset if options['save_all'].lower() == 'yes'.

    @author: Peter Clark
    """
    if 'deformation' in derived_dataset['ds']:
        deformation = derived_dataset['ds']['deformation']
        return deformation

    # Check uvw_names
    if not all([x in source_dataset for x in uvw_names]):
        raise ValueError(f'The u, v, and w variable names, {uvw_names}, \
                           are not all present in the source_dataset passed \
                           to the deformation() function.')

    u = get_data(source_dataset, ref_dataset, uvw_names[0], options)
    [iix, iiy, iiz] = get_string_index(u.dims, ['x', 'y', 'z'])

    sh = np.shape(u)

    max_ch = monc_utils.global_config['chunk_size']

    nch = int(sh[iix]/(2**int(np.log(sh[iix]*sh[iiy]*sh[iiz]/max_ch)/np.log(2)/2)))

    logger.info(f'Deformation nch={nch}')

    u = re_chunk(u, xch=nch, ych=nch, zch = 'all')

    if "z_w" in source_dataset:
        z_w = source_dataset["z_w"]
    elif "z" in source_dataset:
        z_w = source_dataset["z"].rename({'z':'z_w'})
        z_w.name = 'z_w'
        
    if "z_p" in source_dataset:
        z_p = source_dataset["z_p"]
    elif "zn" in source_dataset:
        z_p = source_dataset["zn"].rename({'zn':'z_p'})
        z_p.name = 'z_p'

    ux = do.d_by_dx_field(u, z_w, z_p, grid = grid )

    uy = do.d_by_dy_field(u, z_w, z_p, grid = grid )

    uz = do.d_by_dz_field(u, z_w, z_p, grid = grid )

    del u # Save some memory

    v = get_data(source_dataset, ref_dataset, uvw_names[1], options)
    v = re_chunk(v, xch=nch, ych=nch, zch = 'all')

    vx = do.d_by_dx_field(v, z_w, z_p, grid = grid )

    vy = do.d_by_dy_field(v, z_w, z_p, grid = grid )

    vz = do.d_by_dz_field(v, z_w, z_p, grid = grid )

    del v # Save some memory

    w = get_data(source_dataset, ref_dataset, uvw_names[2], options)
    w = re_chunk(w, xch=nch, ych=nch, zch = 'all')

    wx = do.d_by_dx_field(w, z_w, z_p, grid = grid )

    wy = do.d_by_dy_field(w, z_w, z_p, grid = grid )

    wz = do.d_by_dz_field(w, z_w, z_p, grid = grid )

    del w # Save some memory

    logger.info('Concatenating derivatives')

    t0 = xr.concat([ux, uy, uz], dim='j', coords='minimal',
                   compat='override')
    t1 = xr.concat([vx, vy, vz], dim='j', coords='minimal',
                   compat='override')
    t2 = xr.concat([wx, wy, wz], dim='j', coords='minimal',
                   compat='override')

    defm = xr.concat([t0, t1, t2], dim='i')
    defm.name = 'deformation'
    defm.attrs={'units':'s-1'}

    logger.info(f'Deformation: {defm}')

    if options is not None and options['save_all'].lower() == 'yes':
        defm = save_field(derived_dataset, defm)

    return defm

def shear(d, no_trace:bool=True) :
    r"""
    Compute shear tensor from deformation tensor.

    Shear tensor is defined by
    :math:`S_{ij}={\partial u_{i}}/{\partial {x_j}}
    +{\partial u_{j}}/{\partial {x_i}}`.

    Parameters
    ----------
    d : xarray
        Deformation tensor with dimensions 'i' and 'j'.
    no_trace : bool, optional
        If true, subtract trace (divergence). The default is True.

    Returns
    -------
    S : xarray
        Shear with new dimension 'i_j'.
    mod_S_sq : xarray

        Modulus of shear squared. :math:`1/2 S_{ij}S_{ij}`

    """
    trace = 0
    vname = ''
    if no_trace :
        for k in range(3) :
            trace = trace + d.isel(i=k, j=k)
        # 2 arises because its the trace of S_ij we want.
        trace = (2.0/3.0) * trace
        vname = 'n'

    mod_S_sq = 0

    S = []
    i_j = []
    for k in range(3) :
        for l in range(k,3) :

            S_kl = d.isel(i=k, j=l) + d.isel(i=l, j=k)
            # Note normalisation.
            # We are calculating S_ij = du_i/dx_j + du_j/dx_i
            # |S|^2 = 0.5 S_ij S_ij
            # However We are only calculating ij terms, not ji
            # for efficiency so we include 0.5 S_ji S_ji implicitly
            # by losing the 0.5 in off-diagonal terms.
            # That is, we only sum over half of the off-diagonals
            # and take that to be equivalent to summing all off-diagonals
            # and dividing by two, while only actually applying the half
            # term to the diagnonals to yield 1/2 S_{ij}S_{ij}
            if k == l :
                S_kl = S_kl - trace
                mod_S_sq += 0.5 * S_kl * S_kl
            else :
                mod_S_sq +=  S_kl * S_kl

            S.append(S_kl)
            i_j.append(f'{k:d}_{l:d}')

    S = xr.concat(S, dim='i_j', coords='minimal', compat='override')
    S.coords['i_j'] = i_j
    S.name = 'shear' + vname
    S.attrs={'units':'s-1'}
    S = re_chunk(S)

    mod_S_sq.name = 'mod_S_sq' + vname
    mod_S_sq.attrs={'units':'s-2'}
    S = re_chunk(S)

    return S, mod_S_sq

def vorticity(d):
    r"""
    Compute vorticity vector from deformation tensor.

    vorticity vector is defined as
    :math:`\omega_{k}=({\partial u_{i}}/{\partial {x_j}}
    -{\partial u_{j}}/{\partial {x_i}})`
    with :math:`kij` defined cyclically (i.e. 123, 231, 312).

    Parameters
    ----------
    d : xarray
        Deformation tensor with dimensions 'i' and 'j'.

    Returns
    -------
    vorticity : xarray
        Vorticity with dimension 'i'.

    """
    v_i = []
    for i in range(3) :
        j=(i+1)%3
        k=(i+2)%3
        v_i.append(d.isel(i=k, j=j) - d.isel(i=j, j=k))

    v = xr.concat(v_i, dim='i')
    v.name='vorticity'
    v.attrs={'units':'s-1'}
    v = re_chunk(v)

    return v
