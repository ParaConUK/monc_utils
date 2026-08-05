#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Jul 21 10:47:00 2021

@author: xm904103
"""

import sys

import numpy as np
import xarray as xr

import monc_utils.monc_utils as mu
import monc_utils.io_um.datain as du
from monc_utils.io_um.datain import set_um_datain_options
from monc_utils.data_utils.string_utils import get_string_index

import monc_utils

import dask

import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, 
            format = "<c>{time:HH:mm:ss.SS}</c>"
                  " | <level>{level:<8}</level>"
                  " | <green>{function:<16}</green>"
                  " : <green>{line:<4}</green> : {message}", 
           colorize=True, 
           level="INFO")
    
logger.enable("monc_utils")

xr.set_options(display_max_rows=30)

dask.config.set({"array.slicing.split_large_chunks": True})

test_case = 0

# rootdir = '"C:/Users/paclk/rootdir - University of Reading/"
#rootdir = "C:/Users/xm904103/rootdir - University of Reading/"
rootdir = Path("F:")

if test_case == 0:
    
    set_um_datain_options({'coord_type':'cartesian',
                           'grid_type':'xy_periodic',
                           'ref_is_init_mean':True,
                          })
    
    indir = rootdir / 'Data/UM_CBL/'
    odir = rootdir / 'Data/UM_CBL/test_out/'
    file = 'dc455_100m_L100a_pr000.nc'

    iz = 5
#    iy = 95
    iy = 89
#    it = 0
    it = 5

elif test_case == 1:
    pass

options, update_config = mu.monc_utils_options(config_file=None)

#%%

options['aliases'] = {
    'th':['theta', 'potential_temperature'],
    'p':['pressure'],
    'tracer': ['tracer_rad2'],
    }

options['save_all'] = 'no'

ds = xr.open_dataset(indir / file)
    
print(ds)

fname = 'test_datain'

var_list = [# 'm01s00i700',
            # 'm01s00i702',
            # 'm01s00i704',
            # 'm01s15i002',
            # 'm01s15i003',
            'm01s00i004',
            'm01s00i407',
            'm01s03i025',
            'm01s03i473',
            ]
            
# var_list = ["traj_tracer_xr",
#             "traj_tracer_yr",
#             "traj_tracer_zr",
#             "u_b",
#             "v_b",
#             "theta",
#             ]
#%%
for var_name in var_list:
    # op_var = di.get_data(dataset, ref_dataset, var_name,
    #                      options=options,
    #                      allow_none=True)
    
    op_field = du.get_um_field(ds, var_name)
    print(op_field)
    
    op_data = du.get_um_data(ds, var_name)
    print(op_data)
    
    op_var = du.get_um_data_on_grid(ds, 
                                    var_name,
                                    options=options)
    print(op_var)
    if op_var is None:
        print(f'{var_name:s} not found.')
    else:
        print(op_var.name, op_var.min().values, op_var.max().values,
              op_var.attrs['units'], op_var.shape)
        [itime, iix, iiy, iiz] = get_string_index(op_var.dims, ['time', 'x', 'y', 'z'])
        # if "tracer" in op_var.name:
        #     lev = np.linspace(0,5,51)
        # else:
        #     
        lev =51
        if itime is not None :
            tvar = op_var.dims[itime]
            plot_var = op_var.isel({tvar:it})
        else:
            plot_var = op_var
           
        if iix is not None and iiy is not None:            
            xvar = op_var.dims[iix]
            yvar = op_var.dims[iiy]
            if iiz is not None:
                zvar = op_var.dims[iiz]
                plot_var.isel({yvar:iy, zvar:slice(1,None)}).plot.contourf(
                    figsize=(12,10), levels=lev, x=xvar)
                xplot_var = plot_var.isel({zvar:iz})
            else:
                xplot_var = plot_var
            xplot_var.plot.contourf(figsize=(12,10), levels=lev, x=xvar)

 
plt.show()
