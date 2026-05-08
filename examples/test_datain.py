#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Jul 21 10:47:00 2021

@author: xm904103
"""
import os
import sys

import numpy as np
import xarray as xr

import monc_utils.monc_utils as mu
import monc_utils.io.datain as di
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

#from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
#from dask.diagnostics import ProgressBar

#from dask.distributed import Client
dask.config.set({"array.slicing.split_large_chunks": True})

test_case = 0

# rootdir = '"C:/Users/paclk/rootdir - University of Reading/"
#rootdir = "C:/Users/xm904103/rootdir - University of Reading/"
rootdir = Path("F:")

if test_case == 0:
    config_file = 'config_test_case_0.yaml'
    indir = rootdir / 'traj_data/r_50m_60s_Monsoon3/'
    odir = rootdir / 'traj_data/r_50m_60s_Monsoon3/test_out/'
#    file = 'diagnostics_3d_ts_21600.nc'
#    ref_file = 'diagnostics_ts_21600.nc'
    file = 'diagnostics_3d_ts_16800.nc'
    ref_file = 'diagnostics_ts_16800.nc'
    iz = 40
#    iy = 95
    iy = 89
#    it = 0
    it = 5

elif test_case == 1:
    config_file = 'config_test_case_1.yaml'
    #indir = '/storage/silver/wxproc/xm904103/'
    indir = rootdir / 'CBL/'
    odir = rootdir / 'CBL/'
    #odir = '/storage/silver/wxproc/xm904103/'
    file = 'diagnostics_3d_ts_13200.nc'
    ref_file = None


options, update_config = mu.monc_utils_options(config_file)

#%%

options['aliases'] = {
    'th':['theta', 'potential_temperature'],
    'p':['pressure'],
    'tracer': ['tracer_rad2'],
    }
options['save_all'] = 'no'
# odir = odir / 'test_datain/'

#dir = rootdir+'Git/python/monc_utils/test_data/BOMEX/'
#odir = rootdir+'Git/python/monc_utils/test_data/BOMEX/'

#os.makedirs(odir, exist_ok = True)

#file = 'diagnostics_ts_18000.0.nc'
#ref_file = 'diagnostics_ts_18000.0.nc'
# var_list = [
#     "saturation",
#     "moist_dbdz",
#     "dmoist_bdz",
#     "dbdz",
#     "dbdz_monc",
#     "buoyancy_moist",
#     "buoyancy",
#     "buoyancy_monc",
#     "u",
#     "v",
#     "w",
#     "th",
#     "p",
#     "tracer",
#     "th_L",
#     "th_v",
#     "thref",
#     "T",
#     "th_e",
#     "th_w",
#     "T_w",
#     "m_vapour",
#     "q_vapour",
#     "cloud_fraction",
#     "q_cloud_liquid_mass",
#     "q_total",
#     "m_total",
#     "rh",
#      ]


var_list = [
#     "saturation",
#     "tracer",
    "thref",
    "pref",
    "theta_L",
    "dbydz(theta_L)",
#    "dbydz(teddy)",
    "theta_v",
    "dbydz(theta_v)",
    "th",
    "dbydz(th)",
#     "q_vapour",
    "q_cloud_liquid_mass",
#     "q_total",
      ]

# var_list = [
    # "dbydx(th)",
    # "dbydy(th)",
    # "dbydz(th)",
    # "dbydy(u)",
    # "dbydz(v)",
    # "dbydz(w)",
    # "dbydz(dbydx(dbydy(th)))",
    # "u",
    # "v",
    # "w",
    # "th",
    # "th_v",
    # "th_L",
    # "th_w",
    # "th_s",
    # "th_sw",
    # "p",
    # "dbydx(p)",
    # "dbydy(p)",
    # "dbydz(p)",
    # "dbydz(th_L)",
    # ]

fname = 'test_datain'

dataset = xr.open_dataset(indir / file)

[itime, iix, iiy, iiz] = get_string_index(dataset.dims, ['time', 'x', 'y', 'z'])
timevar = list(dataset.dims)[itime]
xvar = list(dataset.dims)[iix]
yvar = list(dataset.dims)[iiy]
zvar = list(dataset.dims)[iiz]
max_ch = monc_utils.global_config['chunk_size']

nch = np.min([int(dataset.dims[xvar]/(2**int(np.log(dataset.dims[xvar]
                                            *dataset.dims[yvar]
                                            *dataset.dims[zvar]
                                            /max_ch)/np.log(2)/2))),
              dataset.dims[xvar]])

print(f'nch={nch}')

width = dataset.dims[xvar]

dataset.close()

defn = 1
#dataset = xr.open_dataset(indir+file)
#    dataset = xr.open_dataset(dir+file, chunks={timevar: defn,
#                                                'z':'auto', 'zn':'auto'})

dataset = xr.open_dataset(indir / file, chunks={timevar: defn,
                                            xvar:nch, yvar:nch,
                                            'z':'auto', 'zn':'auto'})
print(dataset)
#    ref_dataset = Dataset(dir+ref_file, 'r')
if ref_file is not None:
    ref_dataset = xr.open_dataset(indir / ref_file)
    
    ref_dataset = ref_dataset[['prefn', 'rho', 'rhon', 'thref', ]]
    [itime] = get_string_index(ref_dataset.dims, ['time', ])
    timevar = list(ref_dataset.dims)[itime]
    
    ref_dataset = ref_dataset.isel({timevar:0}).squeeze(drop=True).drop(timevar)
    
    dataset = xr.merge([dataset, ref_dataset])
else:
    ref_dataset = None
    

for var_name in var_list:
    # op_var = di.get_data(dataset, ref_dataset, var_name,
    #                      options=options,
    #                      allow_none=True)
    op_var = di.get_data_on_grid(dataset, 
                                 var_name,
                                 options=options)
    print(op_var)
    if op_var is None:
        print(f'{var_name:s} not found.')
    else:
        print(op_var.name, op_var.min().values, op_var.max().values,
              op_var.attrs['units'], op_var.shape)
        [itime, iix, iiy, iiz] = get_string_index(op_var.dims, ['time', 'x', 'y', 'z'])
        if "tracer" in op_var.name:
            lev = np.linspace(0,5,51)
        else:
            lev =51
        if iix is not None:
            tvar = op_var.dims[itime]
            xvar = op_var.dims[iix]
            yvar = op_var.dims[iiy]
            zvar = op_var.dims[iiz]
            op_var.isel({tvar:it, zvar:iz}).plot.contourf(
                figsize=(12,10), levels=lev, x=xvar)
            op_var.isel({tvar:it, yvar:iy, zvar:slice(1,None)}).plot.contourf(
                figsize=(12,10), levels=lev, x=xvar)

plt.show()

#%%

c = op_var.copy()

c = c.where(op_var >= 1E-5, 0)
c = c.where(op_var <  1E-5, 1)

c.name = 'Cloud'
cf = c.copy()
cf_mean = cf.mean(dim=['x_p','y_p'])

cf_mean.name = 'Cloud Fraction'

#%%

valid_cloud = cf_mean.where(cf_mean > 0, drop=True).compute()

top = [p.dropna(dim='z_p').z_p[-1] for t,p in valid_cloud.groupby('time')]
bot = [p.dropna(dim='z_p').z_p[0] for t,p in valid_cloud.groupby('time')]


ctop = xr.concat(top, dim='time')  
ctop.name = 'Cloud Top'  
cbot = xr.concat(bot, dim='time')    
cbot.name = 'Cloud Base' 

ctop.plot(x='time', label='Cloud Top')
cbot.plot(x='time', label='Cloud Base')

 
plt.show()
