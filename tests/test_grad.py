import numpy as np
import pytest
import xarray as xr

import matplotlib.pyplot as plt

import monc_utils.data_utils.difference_ops as do
#import difference_ops as do
from utils import create_uniform_grid, create_initial_dataset
import monc_utils

monc_utils.global_config['no_dask'] = True

do.set_difference_ops_options({'UM_grid':False, 'xy_periodic':True})
# do.set_difference_ops_options({'UM_grid':True, 'xy_periodic':True})
# do.set_difference_ops_options({'UM_grid':False, 'xy_periodic':False})
# do.set_difference_ops_options({'UM_grid':True, 'xy_periodic':False})


xr.set_options(display_max_rows=100, 
               display_values_threshold=10, 
               display_expand_data_vars=False)

def test_gradients():
    dx = 25.0
    dL = (dx, dx, dx)
    L = (1000.0, 400.0, 100.0)
    
#    gp = create_uniform_grid(dL=dL, L=L)
    

    ds = create_initial_dataset(dL=dL, L=L)
    z_p = ds['z_p']
    z_w = ds['z_w']
    
    # print(ds['fx'])
    f = ds['fx']
    dfx_by_dx_native = do.d_by_dxy_field_native(f, 'x')
    print(f'{dfx_by_dx_native=}')
    # dfx_by_dx_native = do.d_by_dx_field_native(f)
    dfx_by_dx = do.d_by_dx_field(ds['fx'], z_w, z_p )
    print(f'{dfx_by_dx=}')

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfx_by_dx_native.isel(y_p=0, z_p=0).plot.line('x', label='Native', ax=ax1)
    dfx_by_dx.isel(y_p=0, z_p=0).plot.line('o', label='Numeric', ax=ax1)
    ds['dfxdx'].isel(y_p=0, z_p=0).plot.line('-s', label='Analytic', ax=ax1)
    ax1.legend()    
    
    np.testing.assert_allclose(dfx_by_dx, ds['dfxdx'].sel(x_p=dfx_by_dx.x_p.values), 
                               atol = (2 * np.pi * dL[0] / L[0] ) ** 2)
    
    
    dfy_by_dy_native = do.d_by_dxy_field_native(ds['fy'], 'y')
    print(f'{dfy_by_dy_native=}')
    dfy_by_dy = do.d_by_dy_field(ds['fy'], z_w, z_p )
    print(f'{dfy_by_dy=}')

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfy_by_dy_native.isel(x_p=0, z_p=0).plot.line('x', label='Native', ax=ax1)
    dfy_by_dy.isel(x_p=0, z_p=0).plot.line('o', label='Numeric', ax=ax1)
    ds['dfydy'].isel(x_p=0, z_p=0).plot.line('-s', label='Analytic', ax=ax1)
    ax1.legend()
       
    np.testing.assert_allclose(dfy_by_dy, ds['dfydy'].sel(y_p=dfy_by_dy.y_p.values), 
                               atol = (2 * np.pi * dL[1] / L[1] ) ** 2)
    
    f = ds['fz']
    dfz_by_dz_native= do.d_by_dz_field_native(f)
    print(f'{dfz_by_dz_native=}')
    dfz_by_dz= do.d_by_dz_field(ds['fz'], z_w, z_p )
    print(f'{dfz_by_dz=}')

    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))    
    dfz_by_dz_native.isel(x_p=0, y_p=0).plot.line('-x', label='Native', ax=ax1)
    dfz_by_dz.isel(x_p=0, y_p=0).plot.line('-o', label='Num', ax=ax1)
    ds['dfzdz'].isel(x_p=0, y_p=0).plot.line('-s', label='Anal', ax=ax1)
    ax1.legend()
       
    np.testing.assert_allclose(dfz_by_dz, ds['dfzdz'], 
                               atol = 2 * ( dL[2] / L[2]) ** 2)
 
    return    
    
test_gradients()