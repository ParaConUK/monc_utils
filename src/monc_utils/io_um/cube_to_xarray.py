# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:19:16 2024

@author: xm904103
"""
import iris 

import numpy as np
import xarray as xr

from loguru import logger


#from . import um_datain_options
# from monc_utils.io.file_utils import get_full_dim_name
from monc_utils.io_um import ( um_datain_options, 
                               var_properties,
                               stash_map,
                               inverse_lookup,
                             )

from monc_utils.data_utils.string_utils import get_string_index

# from matplotlib import pyplot as plt

Z_COORDS = [
            'level_height',
            'altitude',
            'model_level_number',
            'sigma',
            ]

H_COORDS = [
            'surface_altitude',
            ]

def cube_coord_list(cube):
    return [cd.name() for cd in cube.coords()]

def unique_coords(cubes, coord:str):
    coord_list = []
    for c in cubes:
        cube_coords = cube_coord_list(c)
        if coord not in cube_coords: continue
        cs = c.coord(coord)
        if len(coord_list) == 0: 
            coord_list.append(cs)
        else:
            matched = False
            for s in coord_list:
                if np.array_equiv(s.points, cs.points):
                    matched = True
                    break
            if not matched : coord_list.append(cs)
    return coord_list

def get_z_coords(cubes):
    
    z_rho = z_theta = None
    z_rho_index = z_theta_index = None
    for cube in cubes:
        stash = str(cube.attributes['STASH'])
        name = inverse_lookup(stash, stash_map)
        # print(f'{stash=}, {name=}')
        if name != stash:
            if name in var_properties:
                grid = var_properties[name]['grid']
                coord = cube.coord('level_height')
                # print(f'{grid=}, {coord=}')
                if grid[2]:
                    if z_theta is None or coord.shape[0] > z_theta.shape[0]: 
                        z_theta = coord
                else:
                    if z_rho is None or coord.shape[0] > z_rho.shape[0]: 
                        z_rho = coord
                    
        
    coord_list = unique_coords(cubes, 'level_height')
    
    pop_index = []
    for i, c1 in enumerate(coord_list):
        if c1 == z_theta or c1 == z_rho:
            pop_index.append(i)
            
    if pop_index : 
        for p in sorted(pop_index, reverse=True): coord_list.pop(p)
        
                   
    # for i, c1 in enumerate(coord_list[:-1]):
    for i, c1 in enumerate(coord_list):
        if c1.bounds[0,0] == 0.0 and c1.bounds[1,0] != 0.0:
            if z_rho is None: 
                z_rho = c1
                z_rho_index = i  
            # elif (z_rho != c1 and 
            #       np.allclose(c1.points, np.mean(c1.bounds, axis=1))):
            #     z_rho = c1
            #     z_rho_index = i  
                
        if c1.bounds[0,0] > 0.0:
            if z_theta is None:
                z_theta = c1        
                z_theta_index = i            
            # elif (z_theta != c1 and 
            #      np.allclose(c1.points, np.mean(c1.bounds, axis=1))):
            #     z_theta = c1        
            #     z_theta_index = i     
                
    if z_theta is None:
        if z_rho is not None: 
            if z_rho_index is not None:
                z_theta = np.concatenate([z_rho.bounds[:,0],
                                          np.array([z_rho.bounds[-1,1]])])            
            # else:
            #     for i, c1 in enumerate(coord_list):
            #         if c1 == z_rho:
            #             z_rho_index = i
            #             break
            # if z_rho_index is not None:
                coord_list.pop(z_rho_index)     
            z_rho = z_rho.points
    
    elif z_rho is None:
        if z_theta is not None: 
                # z_rho = np.concatenate([z_theta.bounds[:,0],
                                        # np.array([z_theta.bounds[-1,1]])])
            z_rho = 0.5 * (z_theta.points[1:] + z_theta.points[0:-1])
            if z_theta.points[0] == 0:
                z_rho = np.concatenate([z_rho, 
                                        np.array([z_theta.bounds[-1,1]])])    
            else:
                z_rho = np.concatenate([np.array([0.5 * z_theta.points[0]]), z_rho])
            # else:
            #     for i, c1 in enumerate(coord_list):
            #         if c1 == z_theta:
            #             z_theta_index = i
            #             break
            if z_theta_index is not None:
                coord_list.pop(z_theta_index)     
            z_theta = z_theta.points
    else:
        z_rho = z_rho.points
        z_theta = z_theta.points
                
    # print(z_theta, z_rho)
    
    return z_rho, z_theta, coord_list


def get_horiz_coords(cubes, coord):
    if 'long' in coord: idim = 0
    if 'lat' in coord:  idim = 1
    
    coord_u = coord_p = None
    u_index = p_index = None
    for cube in cubes:
        stash = str(cube.attributes['STASH'])
        name = inverse_lookup(stash, stash_map)
        # print(f'{stash=}, {name=}')
        if name != stash:
            if name in var_properties:
                
                # print(f"Setting grid {coord} from {name}")
                grid = var_properties[name]['grid']
                
                coord_names = cube_coord_list(cube)
                # print(f'{coord=} {coord_names=}')
        
                # print(f'{grid=}, {coord=}')
                [c_index] = get_string_index(coord_names, [coord])
                
                # print(f'{c_index=}')

                if c_index is not None:
                    cube_coord = cube.coord(coord_names[c_index])
                    if grid[idim]:
                        if coord_u is None or cube_coord.shape[0] > coord_u.shape[0]: 
                            coord_u = cube_coord
                            # print(f'Set coord_{["u","v"][idim]} for {idim=} {coord_u.name()}')
                    else:
                        if coord_p is None or cube_coord.shape[0] > coord_p.shape[0]: 
                            coord_p = cube_coord     
                            # print(f'Set coord_p for {idim=} {coord_p.name()}')
                            
                # else:
                #     print(f'No {coord} in {coord_names}.')

    coord_list = unique_coords(cubes, coord)
    
    if coord_u in coord_list: coord_list.remove(coord_u)
    if coord_p in coord_list: coord_list.remove(coord_p)

    if coord_u is None and coord_p is None:
                
        for i,ci in enumerate(coord_list[:-1]):
            for j,cj in enumerate(coord_list[i+1:]):
                cip = ci.points
                cjp = cj.points
                if cjp[0] < cip[0]:
                    cjpm = 0.5 * (cjp[0:-1] + cjp[1:])
                    atol = 0.05 * (cjp[1:] - cjp[0:-1]).max()
                    if np.allclose(cip[0:len(cjpm)], cjpm, atol=atol):
                        coord_u = cjp
                        coord_p = cip
                        
                        # print('Set coord_u to cjp, coord_p to cip')
                        # coord_list.remove(coord_u)
                        # if coord_p in coord_list: coord_list.remove(coord_p)
                elif cip[0] < cjp[0]:
                    cipm = 0.5 * (cip[0:-1] + cip[1:])
                    atol = 0.05 * (cip[1:] - cip[0:-1]).max()
                    if np.allclose(cjp[0:len(cipm)], cipm, atol=atol):
                        coord_u = cip
                        coord_p = cjp
                        # print('Set coord_u to cip, coord_p to cjp')
    else:
        if coord_u is None:
            coord_u = 0.5 * (coord_p.points[0:-1] + coord_p.points[1:])
            coord_u = np.append(coord_u, 
                                np.array([ 2.0 * coord_u[-1] - coord_u[-2]])) 
            if coord_p in coord_list: coord_list.remove(coord_p)
            coord_p = coord_p.points
        else:
            coord_p = 0.5 * (coord_u.points[0:-1] + coord_u.points[1:])
            coord_p = np.append(np.array([ 2.0 * coord_p[0] - coord_p[1]]),
                                coord_p) 
            if coord_u in coord_list: coord_list.remove(coord_u)
            coord_u = coord_u.points

    # if coord_u is None: print(f"No coord_u for dim {idim}")

    # if coord_p is None: print(f"No coord_p for dim {idim}")

    # print(coord_u.min(), coord_p.min())
    return coord_u, coord_p, coord_list


def rename_coords(da, subscript, coords_to_rename):
    rename_dict = {}
    for coord in coords_to_rename:
        if coord in da.coords: 
            rename_dict[coord] = f'{coord}_{subscript}'
            # print(f'Rename {coord} to {coord}_{subscript}')
    
    if rename_dict : da = da.rename(rename_dict)
    return da

def identify_zcoord(cube_zcoord, da, z_rho, z_theta, coord_list):    
    if len(cube_zcoord.points) <= len(z_rho):
        if np.allclose(cube_zcoord.points, z_rho[0:len(cube_zcoord.points)]):
            #da = da.swap_dims({'model_level_number':'level_height'})
            da = rename_coords(da, 'p', Z_COORDS)
            return da
    if len(cube_zcoord.points) <= len(z_theta):
        if np.allclose(cube_zcoord.points, z_theta[0:len(cube_zcoord.points)]):
            #da = da.swap_dims({'model_level_number':'level_height'})
            da = rename_coords(da, 'w', Z_COORDS)
            return da
    for i, c in enumerate(coord_list):
        if cube_zcoord == c:
            # logger.info(f"Renaming {da.name} level_height as z_{i}.")            
            da = da.swap_dims({'model_level_number':'level_height'})
            subscript = f'z{i}'
            if len(cube_zcoord.points) < len(z_rho):
                if np.allclose(cube_zcoord.points, 
                               z_rho[1:len(cube_zcoord.points)+1]):
                    subscript = 'p1'
            if len(cube_zcoord.points) < len(z_theta):
                if np.allclose(cube_zcoord.points, 
                               z_theta[1:len(cube_zcoord.points)+1]):
                    subscript = 'w1'
            da = rename_coords(da, subscript, Z_COORDS)
            break
    return da

def identify_long_coord(cube_long_coord, da, long_u, long_p, coord_list):
    
    minlu = min(len(cube_long_coord.points), len(long_u)) 
    minlp = min(len(cube_long_coord.points), len(long_p)) 
    
    # print(cube_long_coord.name(),
    #       np.max(np.abs(cube_long_coord.points[0:minlu] - long_u[0:minlu])),
    #       np.max(np.abs(cube_long_coord.points[0:minlp] - long_p[0:minlp]))
    #      )
    
    # if len(cube_long_coord.points) <= len(long_p):
    if np.allclose(cube_long_coord.points[0:minlp], long_p[0:minlp]):
        name = cube_long_coord.name()
        # print(f'Renaming {name} {name}_p')
        da = da.rename({name:f'{name}_p'})
        da = rename_coords(da, 'p', H_COORDS)
        return da

    # if len(cube_long_coord.points) <= len(long_u):
    if np.allclose(cube_long_coord.points[0:minlu], long_u[0:minlu]):       
        name = cube_long_coord.name()
        # print(f'Renaming {name} {name}_u')
        da = da.rename({name:f'{name}_u'})
        da = rename_coords(da, 'u', H_COORDS)
        return da
            
    for i, c in enumerate(coord_list):
        if cube_long_coord == c:
            subscript = f'x{i}'
            if np.allclose(cube_long_coord.points[0:minlu-1], 
                           long_u[1:minlu]):
                subscript = 'u1'
            if np.allclose(cube_long_coord.points[0:minlp-1], 
                           long_p[1:minlp]):
                subscript = 'p1'
            da = rename_coords(da, subscript, H_COORDS)
            break
    return da

def identify_lat_coord(cube_lat_coord, da, lat_v, lat_p, coord_list):
    
    tol = 0.2 * (lat_p[1] - lat_p[0])
    
    minlv = min(len(cube_lat_coord.points), len(lat_v)) 
    minlp = min(len(cube_lat_coord.points), len(lat_p)) 

    # print(cube_lat_coord.name(),
    #       np.max(np.abs(cube_lat_coord.points[0:minlv] - lat_v[0:minlv])),
    #       np.max(np.abs(cube_lat_coord.points[0:minlp] - lat_p[0:minlp])),
    #       cube_lat_coord.points[0:2], lat_v[0:2], lat_p[0:2],
    #       tol
    #      )
    
    # plt.plot(cube_lat_coord.points[0:minlv] - lat_v[0:minlv])
    # plt.show()
    
    # plt.plot(cube_lat_coord.points[0:minlp] - lat_p[0:minlp])
    # plt.show()
    
    if np.allclose(cube_lat_coord.points[0:minlp], lat_p[0:minlp], atol = tol):
        name = cube_lat_coord.name()
        # print(f'Renaming {name} {name}_p')
        da = da.rename({name:f'{name}_p'})
        da = rename_coords(da, 'p', H_COORDS)
        return da
    
    if np.allclose(cube_lat_coord.points[0:minlv], lat_v[0:minlv], atol = tol):
        name = cube_lat_coord.name()
        # print(f'Renaming {name} {name}_v')
        da = da.rename({name:f'{name}_v'})
        da = rename_coords(da, 'v', H_COORDS)
        return da
    
    for i, c in enumerate(coord_list):
        if cube_lat_coord == c:
            subscript = f'y{i}'
            if np.allclose(cube_lat_coord.points[0:minlv-1], 
                           lat_v[0:minlv], atol = tol):
                subscript = 'v1'
            if np.allclose(cube_lat_coord.points[0:minlp-1], 
                           lat_p[0:minlp], atol = tol):
                subscript = 'p1'
            da = rename_coords(da, subscript, H_COORDS)
            break
    return da

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


def cubelist_to_dataset(cubes, 
                        offsets:tuple|None=None,
                        out_prec:str = 'float32',
                        ):
    
    match um_datain_options['grid_type'].lower():
        case 'global':
            x_coord = 'longitude'
            y_coord = 'latitude'
        case 'lam':
            x_coord = 'grid_longitude'
            y_coord = 'grid_latitude'
            
    
    coord_list = unique_coords(cubes, 'level_height')
    
    z_rho, z_theta, z_coord_list = get_z_coords(cubes)
    # print(z_rho, z_theta, z_coord_list)
    
    long_u, long_p, long_coord_list = get_horiz_coords(cubes, x_coord)
    
    # print(f'** long_u {np.min(long_u)}, {np.max(long_u)}, {np.shape(long_u)}') 
    # print(f'** long_p {np.min(long_p)}, {np.max(long_p)}, {np.shape(long_p)}') 

    lat_v, lat_p, lat_coord_list = get_horiz_coords(cubes, y_coord)
    # print(f'** lat_v {np.min(lat_v)}, {np.max(lat_v)}, {np.shape(lat_v)}') 
    # print(f'** lat_p {np.min(lat_p)}, {np.max(lat_p)}, {np.shape(lat_p)}') 
    
    # print(z_rho)
    # print(z_theta)
    
    ds = xr.Dataset()
    ds.attrs['grid_type']  = um_datain_options['grid_type'].lower()
    ds.attrs['coord_type'] = um_datain_options['coord_type'].lower()
    
    for cube in cubes:
        
        stash = str(cube.attributes['STASH'])
      
        da = cube_to_dataarray(cube, stash, 
                               offsets=offsets,
                               out_prec=out_prec, 
                               )
        coord_names = cube_coord_list(cube)
        
        if 'level_height' in coord_names:
            cube_zcoord = cube.coord('level_height')
            da = identify_zcoord(cube_zcoord, da, 
                                 z_rho, z_theta, coord_list)    

        [long_index] = get_string_index(coord_names, ['longitude'])
        if long_index is not None:
            cube_long_coord = cube.coord(coord_names[long_index])
            # print(f'{cube_long_coord=}')
            da = identify_long_coord(cube_long_coord, da, 
                                     long_u, long_p, long_coord_list)    

        [lat_index] = get_string_index(coord_names, ['latitude'])
        if lat_index is not None:
            cube_lat_coord = cube.coord(coord_names[lat_index])
            # print(f'{cube_lat_coord=}')
            da = identify_lat_coord(cube_lat_coord, da, 
                                    lat_v, lat_p, lat_coord_list)    
        
        ds[stash] = da        
            
        logger.info(f"Cube {cube.name()} STASH {stash} added to output as {da.name}.")
    
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