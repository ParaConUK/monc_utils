# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:27:25 2018

@author: Peter Clark
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask

import monc_utils.data_utils.deformation as defm


from monc_utils.data_utils.string_utils import get_string_index
#from subfilter.io.dataout import save_field
from monc_utils.io.datain import get_data_on_grid
from monc_utils.io.dataout import setup_child_file

from monc_utils.io.file_utils import configure_model_resolution
                                 

import monc_utils
test_case = 0

override = True

plot_type = '.png'

def main():
    """Top level code, a bit of a mess."""
    if test_case == 0:
        config_file = 'config_test_case_0.yaml'
        indir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
        odir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
        file = 'diagnostics_3d_ts_21600.nc'
        ref_file = 'diagnostics_ts_21600.nc'
    elif test_case == 1:
        config_file = 'config_test_case_1.yaml'
        indir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
        odir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
        file = 'diagnostics_3d_ts_13200.nc'
        ref_file = None

    fname = 'test_deformation_org'

    odir = odir + fname +'/'
    os.makedirs(odir, exist_ok = True)

    plot_dir = odir + 'plots/'
    os.makedirs(plot_dir, exist_ok = True)

    # Avoid accidental large chunks and read dask_chunks
    if not monc_utils.global_config['no_dask']:
        dask.config.set({"array.slicing.split_large_chunks": True})
        dask_chunks = monc_utils.global_config['dask_chunks']

    # Read data
    dataset = xr.open_dataset(indir+file, chunks=dask_chunks)

    print(dataset)

    if ref_file is not None:
        ref_dataset = xr.open_dataset(indir+ref_file)
    else:
        ref_dataset = None

    # Get model resolution values
    dx, dy, options = configure_model_resolution(dataset)

    [itime, iix, iiy, iiz] = get_string_index(dataset.dims, ['time', 'x', 'y', 'z'])
    [timevar, xvar, yvar, zvar] = [list(dataset.dims)[i] for i in [itime, iix, iiy, iiz]]

    npoints = dataset.dims[xvar]

# For plotting
#    ilev = 15
    ilev = 40
#    iy = 40
    iy = 95

    opgrid = 'w'

    derived_data, exists = setup_child_file( indir+file, odir, fname,
                                   options, override=override)
    if exists :
        print('Derived data file exists' )
        print("Variables in derived dataset.")
        print(derived_data['ds'].variables)

    d = defm.deformation(dataset, ref_dataset, derived_data,
                options, grid=opgrid)
    
    S_ij, mod_S = defm.shear(d, no_trace=False)
    
    z = dataset["z"]
    zn = dataset["zn"]


    print("Plotting mod_S")
    plot_shear(mod_S, z, plot_dir, ilev, iy,
                no_trace = False)
    print('--------------------------------------')

    print(derived_data)

    print('--------------------------------------')
    derived_data['ds'].close()
    dataset.close()


def plot_shear(var, zcoord, plot_dir, ilev, iy, no_trace = True):
    var_name = var.name
    if no_trace : var_name = var_name+'n'

    [iix, iiy, iiz] = get_string_index(var.dims, ['x', 'y', 'z'])
    [xvar, yvar, zvar] = [list(var.dims)[i] for i in [iix, iiy, iiz]]

    for it, time in enumerate(var.coords['time']):
        print(f'it:{it}')

        pltdat = var.isel(time=it)

        nlevels = 40
        plt.clf

        fig1, axa = plt.subplots(3,1,figsize=(10,12))

        Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0], levels=nlevels)

        Cs3 = pltdat.isel({yvar:iy, zvar:slice(1,None,None)}).plot.imshow(x=xvar, y=zvar, ax=axa[1], levels=nlevels)

        p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2])

        plt.tight_layout()

        plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy
                    +'_%02d'%it+plot_type)
        plt.close()

    return
if __name__ == "__main__":
    main()
