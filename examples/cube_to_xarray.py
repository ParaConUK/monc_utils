# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:19:16 2024

@author: xm904103
"""
import iris 

import xarray as xr
from monc_utils.io_um.datain import (coords_to_latlon, coords_to_cartesian)

from loguru import logger

def cube_to_dataarray(cube, 
                      name:str|None,
                      offsets:tuple|None=None,
                      out_prec:str = 'float32',
                      ):
    """
    Convert iris.Cube

    Parameters
    ----------
    cube : iris.Cube
    name : str|None
        Name of output DataArray.
        If None STASH string is used. 
    out_prec : str, optional
        Precision to use. The default is 'float32'.

    Returns
    -------
    da : TYPE
        DESCRIPTION.

    """
    
    da = xr.DataArray.from_iris(cube)
    
    if  name is not None:               
        da.name = name
    else:
        da.name = str(cube.attributes['STASH'])
    
        
    if da.dtype != out_prec:
        da = da.astype(out_prec)
        
    return da

def iris_files_to_dataarray(files, stash, name, out_prec:str = 'float32',
                            offsets:tuple|None=None):
    logger.info(f"Reading cube from file {files}.")
    
    constraint = iris.NameConstraint(STASH=stash)
    try:
        cube = iris.load_cube(files, constraint)
    except iris.exceptions.ConstraintMismatchError :
        logger.info(f"Cube STASH {stash} not available.")
        cube = None
    if cube is not None:
        
        da = cube_to_dataarray(cube, name, 
                               out_prec=out_prec, 
                               offsets=offsets)
        
        da.name = name
        
        da.attrs['STASH'] = stash
                 
        logger.info(f"Cube {cube.name()} STASH {stash} retrieved as {da.name}.")
   
    return da



def iris_files_to_dataset(files, stash_list:dict, out_prec:str = 'float32',
                          xy_periodic:bool=False,
                          offsets:tuple|None=None):
    logger.info(f"Reading cubes from file {files}.")

    da_list = []
    # da_dict = {}
    for stash, name in stash_list.items():
        constraint = iris.NameConstraint(STASH=stash)
        try:
            cube = iris.load_cube(files, constraint)
        except iris.exceptions.ConstraintMismatchError :
            logger.info(f"Cube {name} STASH {stash} not available.")
            cube = None
        if cube is not None:
            
            da = cube_to_dataarray(cube, name, out_prec=out_prec, 
                                   offsets=offsets)
            
            # da.name = name
            
            da.attrs['STASH'] = stash
            
            print(da)
                
            # da_dict[da.name] = da
            da_list.append(da)
            
            logger.info(f"Cube {cube.name()} STASH {stash} added to output as {da.name}.")
            
    ds = xr.merge(da_list, compat='override')
    ds.attrs["xy_periodic"] = xy_periodic

    return ds
     
def cubelist_to_dataset(cubes, 
                        offsets:tuple|None=None,
                        out_prec:str = 'float32',
                        ):
    da_list = []
    # da_dict = {}

    for cube in cubes:
        
        stash = str(cube.attributes['STASH'])
        
        da = cube_to_dataarray(cube, stash, 
                               offsets=offsets,
                               out_prec=out_prec, 
                               )
            
        # da_dict[da.name] = da
        da_list.append(da)
        
        logger.info(f"Cube {cube.name()} STASH {stash} added to output as {da.name}.")

    ds = xr.merge(da_list, compat='override')
    # ds = xr.merge(da_dict)
    
    return ds

def open_um_dataset_iris(input_files: str, 
                         orog_file: str, 
                         offsets: dict|None = None, 
                         xy_periodic:bool=False, 
                         out_prec:str = 'float32'):
    """
    Read files with iris to return dataset with variables in list given by 
    stash_list.

    Parameters
    ----------
    input_files : str
        Filespec including wildcards for input iris-compatible data.
    orog_file : str
        Filespec for input iris-compatible data containg orography.
    stash_list : dict
        Mapping of STASH numbers (string of form m01sxxiyyy) to variable names.
        If variable name is None, use the STASH string.
    offsets : tuple
        Integer offsets to add to 'x', 'y' and 'z' index coordinates.
    xy_periodic : bool, optional
        DESCRIPTION. The default is False.
    out_prec : str, optional
        DESCRIPTION. The default is 'float32'.

    Returns
    -------
    ds : xr.Dataset

    """
    
    logger.info("Reading input cubes.")
            
    cubes = iris.load([input_files, orog_file])
    
    ds = cubelist_to_dataset(cubes, 
                             offsets=offsets, 
                             out_prec=out_prec)

    # simulations with MONC always have periodic boundary conditions
    ds.attrs["xy_periodic"] = xy_periodic

    return ds