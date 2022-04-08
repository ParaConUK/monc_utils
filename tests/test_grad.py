import numpy as np
import pytest
import xarray as xr

import matplotlib.pyplot as plt

import monc_utils.data_utils.difference_ops as do
#import difference_ops as do
from utils import create_uniform_grid, create_initial_dataset
import monc_utils

monc_utils.global_config['no_dask'] = True


def test_gradients():
    dx = 25.0
    dL = (dx, dx, dx)
    L = (1000.0, 400.0, 100.0)
    
#    gp = create_uniform_grid(dL=dL, L=L)
    

    ds = create_initial_dataset(dL=dL, L=L)
    z_p = ds['z_p']
    z_w = ds['z_w']
    
    # print(ds['fx'])
    
    dfx_by_dx = do.d_by_dx_field(ds['fx'], z_w, z_p )

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfx_by_dx.isel(y_p=0, z_p=0).plot.line('-o', label='Num', ax=ax1)
    ds['dfxdx'].isel(y_p=0, z_p=0).plot.line('-s', label='Anal', ax=ax1)
    ax1.legend()    
    
    np.testing.assert_allclose(dfx_by_dx, ds['dfxdx'], 
                               atol = (2 * np.pi * dL[0] / L[0] ) ** 2)
    
    
    dfy_by_dy = do.d_by_dy_field(ds['fy'], z_w, z_p )

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfy_by_dy.isel(x_p=0, z_p=0).plot.line('-o', label='Num', ax=ax1)
    ds['dfydy'].isel(x_p=0, z_p=0).plot.line('-s', label='Anal', ax=ax1)
    ax1.legend()
       
    np.testing.assert_allclose(dfy_by_dy, ds['dfydy'], 
                               atol = (2 * np.pi * dL[1] / L[1] ) ** 2)
    
    
    dfz_by_dz= do.d_by_dz_field(ds['fz'], z_w, z_p )

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfz_by_dz.isel(x_p=0, y_p=0).plot.line('-o', label='Num', ax=ax1)
    ds['dfzdz'].isel(x_p=0, y_p=0).plot.line('-s', label='Anal', ax=ax1)
    ax1.legend()
       
    np.testing.assert_allclose(dfz_by_dz, ds['dfzdz'], 
                               atol = 2 * ( dL[2] / L[2]) ** 2)
 
    return    
    
test_gradients()