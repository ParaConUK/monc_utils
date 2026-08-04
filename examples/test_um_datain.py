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
from monc_utils.io_um.cube_to_xarray import cubelist_to_dataset

import monc_utils

import iris 


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

test_case = 2

# rootdir = '"C:/Users/paclk/rootdir - University of Reading/"
#rootdir = "C:/Users/xm904103/rootdir - University of Reading/"
rootdir = Path("F:")

if test_case == 0:
    
    set_um_datain_options({'coord_type':'rotated_lon_lat',
                           'grid_type':'lam',
                           'ref_is_init_mean':True,
                          })
    
    indir = rootdir / 'Trajectory_Tests/nsaa_10min_v2/'
    odir =  rootdir / 'Trajectory_Tests/nsaa_10min_v2/test_out/'
    file = 'umnsaa_pc004'
    orog_file = 'umnsaa_pa000'
    
    files = [indir / file, indir / orog_file]
    offsets = {'x':0, 'y':0, 'z':1}

    iz = 15
#    iy = 95
    iy = 89
#    it = 0
    it = 5

    var_list = ['m01s00i700',
                # 'm01s00i702',
                # 'm01s00i704',
                'm01s15i002',
                # 'm01s15i003',
                # 'm01s00i004',
                # 'm01s03i216',
                ]

elif test_case == 1:
    
    set_um_datain_options({'coord_type':'lon_lat',
                           'grid_type':'global',
                           'ref_is_init_mean':True,
                          })

    indir = rootdir / 'Trajectory_Tests/glm_nolev0/'
    odir =  rootdir / 'Trajectory_Tests/glm_nolev0/'
    
    file = '20230831T0030Z_glm_pc000.pp'
    
    files = indir / file
    offsets = {'x':0, 'y':0, 'z':1}

    iz = 15
#    iy = 95
    iy = 89
#    it = 0
    it = 5
    var_list = ['m01s00i700',
                # 'm01s00i702',
                # 'm01s00i704',
                'm01s00i150',
               ]

elif test_case == 2:
    
    set_um_datain_options({'coord_type':'lon_lat',
                           'grid_type':'global',
                           'ref_is_init_mean':True,
                          })

    indir = rootdir / 'Trajectory_Tests/glm/'
    odir =  rootdir / 'Trajectory_Tests/glm/'
    
    file = '20230831T0030Z_glm_pc*.pp'
    
    files = indir.glob(file)
    offsets = {'x':0, 'y':0, 'z':1}

    iz = 15
#    iy = 95
    iy = 89
#    it = 0
    it = 19
    var_list = ['m01s00i700',
                'm01s00i701',
                'm01s00i702',
                'm01s00i704',
                'm01s00i150',
               ]


options, update_config = mu.monc_utils_options(config_file=None)

#%%

options['aliases'] = {
                    'th':['theta', 'potential_temperature'],
                    'p':['pressure'],
                    'tracer': ['tracer_rad2'],
                     }

options['save_all'] = 'no'

   
    
cubes = iris.load(files)

# cubes = cubes[:4] 

ds = cubelist_to_dataset(cubes, 
                         offsets=offsets,
                         out_prec='float32',
                         )
    
print(ds)
# print(cubes)

# get_vars = {str(cube.attributes['STASH']):str(cube.attributes['STASH']) 
#             for cube in cubes}


# for key in get_vars:
#     print(key)

fname = 'test_datain'

#dataset = xr.open_dataset()

#get_vars = {'m01s00i700':"traj_tracer_xr",
#            'm01s00i702':"traj_tracer_yr",
#            'm01s00i704':"traj_tracer_zr"}
# ds = iris_files_to_dataset([indir / file, indir / orog_file],
#                             get_vars,
#                             xy_periodic=False,
#                             offsets=offsets)

# dataset = ds

# print(dataset)
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
            
        print(plot_var.name, plot_var.min().values, plot_var.max().values,
              plot_var.attrs['units'], plot_var.shape)
    
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
