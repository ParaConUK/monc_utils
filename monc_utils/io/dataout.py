# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:02:20 2021.

   @author: Peter Clark
"""
import time
import os
import xarray as xr
from dask.diagnostics import ProgressBar

import monc_utils  # for global prameters

def save_field(dataset, field, write_to_file=True):
    """
    Save dask-chunked xarray field to xarray Dataset.

    Parameters
    ----------
    dataset : xarray Dataset
        Output dataset.
    field :  dask-chunked xarray
        Input field.
    write_to_file : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    fn = dataset['file'].split('/')[-1]
    if field.name not in dataset['ds']:
        print(f"Saving {field.name} to {fn}")
        dataset['ds'][field.name] = field
        encoding = {field.name: {"dtype": monc_utils.global_config['output_precision']} }
        if write_to_file:
            d = dataset['ds'][field.name].to_netcdf(
                    dataset['file'],
                    unlimited_dims="time",
                    mode='a', compute=False, encoding = encoding)
            # Toggle ProgressBar depending on compute space
            #   (for cleaner stdout on cluster)
            if monc_utils.executing_on_cluster:
                results = d.compute()
            else:
                with ProgressBar():
                    results = d.compute()
            # This wait seems to be needed to give i/o time to flush caches.
            time.sleep(monc_utils.global_config['write_sleeptime'])
    else:
        print(f"{field.name} already in {fn}")
#    print(dataset['ds'])
    return dataset['ds'][field.name]

def setup_child_file(source_file, destdir, outtag, options, override=False) :
    """
    Create NetCDF dataset for derived data in destdir.

    File name is original file name concatenated with filter_def.id.

    Parameters
    ----------
    source_file     : str
        Input NetCDF file name.
    destdir         : str
        Output directory.
    options         : dict
        Options dictionary
    override=False  : bool
        if True force creation of file

    Returns
    -------
    do              : dict
        {**'file'**: derived_dataset_name (str) - file name,\n
        **'ds'**: derived_dataset (xarray Dataset) - NetCDF dataset for derived data}
    exists          : bool
        True when input **source_file** already existed and was not overwritten

    """
    if os.path.isfile(source_file):
        ds = xr.open_dataset(source_file)
        atts = ds.attrs
    else:
        raise FileNotFoundError(f"Cannot find file {source_file}.")

    derived_dataset_name = os.path.basename(source_file)
    derived_dataset_name = ('.').join(derived_dataset_name.split('.')[:-1])

    if monc_utils.global_config['l_slurm_job_tag'] and monc_utils.executing_on_cluster:
        jn = os.environ['SLURM_JOB_NAME']
        derived_dataset_name = destdir+derived_dataset_name + "_"+jn+ "_" + outtag + ".nc"
    else:
        derived_dataset_name = destdir+derived_dataset_name + "_" + outtag + ".nc"

    exists = os.path.isfile(derived_dataset_name)

    if exists and not override :

        derived_dataset = xr.open_dataset(derived_dataset_name)

    else :
        if exists:
            print(f"Overwriting file {derived_dataset_name}.")
        exists = False

        derived_dataset = xr.Dataset(coords =
                        {'z':ds.coords['z'],'zn':ds.coords['zn']})

        # Ensure bool, dict, and None items can be stored
        atts_out = {**atts, **monc_utils.global_config, **options}
        for inc in atts_out:
            if isinstance(atts_out[inc], (dict, bool, type(None))):
                atts_out[inc] = str(atts_out[inc])
                print(atts_out[inc])
        derived_dataset.attrs = atts_out

        derived_dataset.to_netcdf(derived_dataset_name, mode='w')
        print(derived_dataset)
    do = {'file':derived_dataset_name, 'ds': derived_dataset}
    ds.close()
    return do, exists
