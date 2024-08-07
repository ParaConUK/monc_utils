# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:02:20 2021.

   @author: Peter Clark and Todd Jones
"""
import time
import os
import xarray as xr
from dask.diagnostics import ProgressBar
from pathlib import Path

import monc_utils  # for global prameters

from loguru import logger

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
    fn = Path(dataset['file']).name
    if field.name not in dataset['ds']:
        out_prec = monc_utils.global_config['output_precision']
        if field.dtype != out_prec:
            field = field.astype(out_prec)
        dataset['ds'][field.name] = field
        encoding = {field.name: {"dtype": out_prec}}
        if write_to_file:
            logger.info(f"Saving {field.name} to {fn} with precision {out_prec}")
            d = dataset['ds'][field.name].to_netcdf(
                    dataset['file'],
                    unlimited_dims="time",
                    mode='a', compute=False, encoding = encoding)
            # Toggle ProgressBar depending on compute space
            #   (for cleaner stdout on cluster)
            if monc_utils.executing_on_cluster:
                results = d.compute()
            else:
                print(f"Saving {field.name} to {fn}")
                with ProgressBar():
                    results = d.compute()
            # This wait seems to be needed to give i/o time to flush caches.
            time.sleep(monc_utils.global_config['write_sleeptime'])
    else:
        logger.info(f"{field.name} already in {fn}")
    return dataset['ds'][field.name]

def setup_child_file(source_file, destdir, outtag, options=None, 
                     override=False, keep_coords=None ) :
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
    keep_coords=None     : dict | None
        coordinates to use for child file

    Returns
    -------
    do              : dict
        {**'file'**: derived_dataset_name (str) - file name,\n
        **'ds'**: derived_dataset (xarray Dataset) - NetCDF dataset for derived data}
    exists          : bool
        True when input **source_file** already existed and was not overwritten

    """
    
    source_path = Path(source_file)
    if source_path.is_file():
        ds = xr.open_dataset(source_path)
        atts = ds.attrs
    else:
        raise FileNotFoundError(f"Cannot find file {source_path}.")
        
    if options is None:
        options = {}
        
    if keep_coords is None:
        clist = ['z','zn']
        keep_coords = {c:ds.coords[c] for c in clist if c in ds.coords }
     
    derived_dataset_name = source_path.stem

    if monc_utils.global_config['l_slurm_job_tag'] \
        and monc_utils.executing_on_cluster:
        jn = os.environ['SLURM_JOB_NAME']
        derived_dataset_name = derived_dataset_name + "_" + jn

    derived_dataset_name = destdir+derived_dataset_name \
            + "_" + outtag + ".nc"
            
    derived_dataset_path = Path(derived_dataset_name)
            
    exists = derived_dataset_path.is_file()

    if exists and not override :

        derived_dataset = xr.open_dataset(derived_dataset_name)

    else :
        if exists:
            logger.info(f"Overwriting file {derived_dataset_name}.")
        exists = False

        derived_dataset = xr.Dataset(coords = keep_coords)

        # Ensure bool, dict, and None items can be stored
        atts_out = {**atts, **monc_utils.global_config, **options}
        for inc in atts_out:
            if isinstance(atts_out[inc], (dict, bool, type(None))):
                atts_out[inc] = str(atts_out[inc])
                logger.debug(f'{inc}: {atts_out[inc]}')
        derived_dataset.attrs = atts_out

        derived_dataset.to_netcdf(derived_dataset_name, mode='w')
    do = {'file':derived_dataset_name, 'ds': derived_dataset}
    ds.close()
    return do, exists
