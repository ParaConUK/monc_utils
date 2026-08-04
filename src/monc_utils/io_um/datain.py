"""
Created on Mon Aug  2 11:01:11 2021.

@author: Peter Clark
"""
import numpy as np
import xarray as xr
import typing
import datetime
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.file_utils import get_full_dim_name
from monc_utils.io.datain import (set_coord_type,
                                  set_grid_type,
                                  add_grid_attrs,
                                  clean_coords,
                                  correct_grid_and_units, 
                                  get_derived_vars,
                                  get_derivative)

from monc_utils.io.dataout import save_field
from monc_utils.data_utils.dask_utils import re_chunk

import monc_utils.data_utils.difference_ops as do
import monc_utils.thermodynamics.thermodynamics as th
import monc_utils

from monc_utils.io_um import ( um_datain_options, 
                               var_properties,
                               stash_map,
                               PGRID, WGRID, UGRID, VGRID, TGRID,
                               inverse_lookup,
                             )
               


from loguru import logger

import re

stash_pattern = re.compile("^m01s[0-9]{2}i[0-9]{3}$")

"""
Hierarchy

get_um_data_on_grid
    get_um_and_transform
        get_um_data
            get_um_field
                coords_to_cartesian
                coords_to_latlon
                clean_coords
            get_derived_vars
                get_um_data
            get_derivative
                get_data
                d_by_dx_field_native
                d_by_dy_field_native
                d_by_dz_field_native
            correct_grid_and_units

"""



def set_um_datain_options(opts:dict):
    global um_datain_options
    um_datain_options.update(opts)
    logger.info(f'{um_datain_options=}')
    return
    
def set_um_stashmap(stash_map_update:dict):
    global stash_map
    stash_map.update(stash_map_update)
    logger.info('Updated stash_map.')
    return

def get_grid_type(field, var_name):

    in_grid_type = TGRID
    units = ''
    
    for i, coord in enumerate(['longitude','latitude','model_level_number']):
        c = get_full_dim_name(field, coord)
        if c is not None :
            
            match c[-2:]:
                case '_p':
                    in_grid_type[i] = PGRID[i]
                case '_w':
                    in_grid_type[i] = WGRID[i]
                case '_u':
                    in_grid_type[i] = UGRID[i]
                case '_v':
                    in_grid_type[i] = VGRID[i]
        elif var_name in var_properties:
            in_grid_type[i] = var_properties[var_name]['grid'][i]
            units = var_properties[var_name]['units']
    return in_grid_type, units


def coords_to_latlon(field:xr.DataArray, offsets:typing.Optional[dict]=None):
    """
    Identify coordinates in Lat/Long (or rotated Lat/Long) grid, 
    create dimension coordinates `x_p`, `x_u`, `y_p`, `y_v`, `z_p` or `z_w`
    and modify original (now non-dimension) coordinates to indicate 
    u, v, w, or p grid.
    In the horizontal add coordinate values in terms of grid index, taking 
    account of staggering (so `x_u` starts at -0.5). 
    
    The optional input `offsets` is intended for data cut out of a larger grid
    (e.g. the inner part of a variable resolution grid) and provides an offset 
    in each direction added to the grid index.
    
    The code looks for coordinates containing 'longitude','latitude' and
    'model_level'. It uses the field.name and internal lookup table 
    `var_properties` to detrmine the grid.
    
    Parameters
    ----------
    field : xr.DataArray
        DataArray to be modified.
    offsets : dict, optional
        Numeric offsets added to grid index. The default is None. 
        Keys are 'x', 'y', 'z']

    Returns
    -------
    field : TYPE
        DESCRIPTION.

    """  
       
    if offsets is None:
        offsets={'x':0, 'y':0, 'z':0}
    
    var_name = field.name
    
    in_grid_type, units = get_grid_type(field, var_name)
            
    swap_map = {}
    for i, (coord, alt_point) in enumerate(
            zip(['longitude','latitude','model_level_number'],
                ['u', 'v', 'w'])):
        
        if i > len(in_grid_type)-1: break
        
        new_name = c = get_full_dim_name(field, coord)
        
        # print(f'full_dim_name {coord} {c}')
        
        if c is None and coord == 'level_height':
            new_name = c = get_full_dim_name(field, 'model_level')
        
        if c is None : continue
        
                    
        if coord == 'level_height':
            base_coord_vals = field.coords[c].values.astype("float32")
        else:
            base_coord_vals = np.arange(field.sizes[c], dtype="float32")
            
        
        if in_grid_type[i]:
            if c[-2:] != f'_{alt_point}':
                new_name = f'{c}_{alt_point}'
            coord_vals = base_coord_vals + offsets['xyz'[i]] + [1, 1, 0.0][i]
            new_coord = 'xyz'[i] + f'_{alt_point}'
            
        else:
            
            if c[-2:] != '_p':
                new_name = f'{c}_p'
            coord_vals = base_coord_vals + offsets['xyz'[i]] + [0.5, 0.5, +0.5][i]           
            new_coord = 'xyz'[i] + '_p'
            
        if c != new_name:    
            field = field.rename({c:new_name})
            c = new_name
                 
        nc = {new_coord: (c, coord_vals)}
            
        # print(field)
        # print(f'{nc=}')
        field = field.assign_coords(nc)
        swap_map[new_name] = new_coord

    field = field.swap_dims(swap_map)
    
    for c in 'xy':        
        field = add_grid_attrs(field, c, grid_type=field.attrs['grid_type'])

    if 'units' not in field.attrs:        
        field.attrs['units'] = units
                  
    return field
    
def coords_to_cartesian(field):
    """
    SPECIFIC TO UM

    Parameters
    ----------
    field : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    def _adjust_cyclic_data_order(field, dimname, vname):
        logger.info(f"Rolling {vname} data")
        field = field.roll({dimname:-1}, roll_coords=True) 
        c = field.coords[old_coord].values
        c[-1] = c[-2]*2 - c[-3]
        field = field.assign_coords({old_coord:c})
        return field
    
    xy_periodic = (field.attrs['grid_type'] == 'xy_periodic')
    
    units = ''

#    field.attrs['cartesian'] = True
#    field, xy_periodic = is_xy_periodic(field, 
#                           xy_periodic_def=
#                           um_datain_options.get('xy_periodic', True))
    
    for new_coord, old_coord in zip(('x', 'y', 'z', 'time'), 
                                    get_um_coords(field)):
        
        if old_coord is None : continue
    
        new_coord_full = None
    
        if new_coord == 'z':
            
            # vert_dim = get_full_dim_name(field, 'lev_eta')
            vert_dim = get_full_dim_name(field, 'eta')
            if vert_dim is not None:
                if 'rho' in vert_dim: 
                    new_coord_full = 'z_p'
                else: 
                    new_coord_full = 'z_w'
            else: 
                vert_dim = get_full_dim_name(field, 'level_number')
                if vert_dim is not None:
                    if 'w' in vert_dim: 
                        new_coord_full = 'z_w'
                    else: 
                        new_coord_full = 'z_p'
                        
            if new_coord_full is None: continue
                
            # print(f'{field=} \n {old_coord} {new_coord_full}')
            field = field.rename({old_coord:new_coord_full})
            field = field.swap_dims({vert_dim:new_coord_full})
            
        else:
            
            new_coord_full = f'{new_coord}{get_um_grid_desc(old_coord)}'
            if new_coord in 'xy':                        
                
                if old_coord == 'longitude_cu':
                    field = _adjust_cyclic_data_order(field, old_coord, 'u')
                    new_coord_full = 'x_u'
                    
                elif old_coord == 'latitude_cu': 
                    new_coord_full = 'y_p'
                    
                elif old_coord == 'longitude_cv':
                    new_coord_full = 'x_p'
                    
                elif old_coord == 'latitude_cv':
                    field = _adjust_cyclic_data_order(field, old_coord, 'v')
                    new_coord_full = 'y_v'
                    
                if field.attrs['grid_mapping'] == 'grid_crs':
                    new_coord_values = np.round(field.coords[old_coord].values  * 1000)
                else:
                    new_coord_values = field.coords[old_coord].values 
                    
                field = field.assign_coords(
                    {new_coord_full: (old_coord, new_coord_values)}) 
                   
                field = field.swap_dims({old_coord:new_coord_full})
                
                field = add_grid_attrs(field, 
                                       new_coord_full, 
                                       grid_type='xy_periodic')
                
            else:
                field = field.rename({old_coord:new_coord_full})          
            
        if new_coord == 'time':    
            hours = (field[new_coord_full].values 
                   - field[new_coord_full].values[0]
                    ) / datetime.timedelta(hours=1)
            
            field = field.assign_coords(
                {'elapsed_time':(new_coord_full, hours.astype(np.float32))})

        if 'bounds' in field[new_coord_full].attrs:
            field[new_coord_full].attrs.pop('bounds')
            
    if 'units' not in field.attrs:        
        field.attrs['units'] = units
        
    return field
    
def get_um_field(ds, name:str, add_name=True):
    """
    Read DataArray corresponding to stash_code from xarray dataset,
    Changing coordinates to more MONC-like.

    Parameters
    ----------
    ds : xarray Dataset
        Input (at least 2D) data.
    name : str
        Actual variable name
    stash : str
        Of form 'm01snnimmm' with nn = section and mmm item.

    Returns
    -------
    field : xarray.core.dataarray.DataArray
        Required data.

    """
    
    grid_type = um_datain_options.get('grid_type', 'lam')
    match grid_type.lower():
        case 'xy_periodic':
            coord_type = 'cartesian'
        case 'lam' :
            coord_type = 'rotated_lon_lat'
        case 'global':
            coord_type = 'lon_lat'
        case _ :
            raise ValueError(f'Grid type {grid_type} unknown.')
            
    cartesian = (coord_type == 'cartesian')  
    
    keep_coords = um_datain_options.get('keep_coords', [])
    
    if name in ds.data_vars: 
        var_name = name
    elif f'STASH_{name}' in ds.data_vars:
        var_name = f'STASH_{name}'
    else:
        raise ValueError(f'field id {name} not in dataset.')
        
    field = ds[var_name]

    if re.match(stash_pattern, name) and add_name:
       field.name = inverse_lookup(name, stash_map)
 
    field = set_grid_type(field, grid_type_def=grid_type) 
    field = set_coord_type(field, coord_type_def=coord_type)              

    # print(field)            
    if cartesian:
        field = coords_to_cartesian(field)
    else:
        field = coords_to_latlon(field)
                 
    # field = clean_coords(field, keep_coords=keep_coords)
    
    return field
    
def get_um_data(source_dataset,
                var_name: str,
                options: dict=None,
                allow_none: bool=False,
                add_name=True) :
    """
    Extract data or derived data field from source NetCDF dataset.

    If var_name is in source_dataset it is retrieved; if one of the primary
    variables with a key in var_properties the grid is corrected.
    Otherwise, it is assumed to be on a 'theta' point.

    Currently written for UM data, enforcing C-grid. Returned coords are
    'x_p', 'x_u', 'y_p', 'y_v', 'z_w', 'z_p'.  

    Alternative names of variables can be supplied in options['aliases'] as
    a list of strings. If var_name is not found in source_dataset the first
    alias present in source_dataset is retrieved and renamed to var_name.

    Currently supported derived data are specified if the thermodynamics module.

    The special var_name 'thref' retrieves the reference theta profile.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    var_name : str
        Name of variable to retrieve.
    options : dict (optional - default=None)
        Options possibly used are 'dx' and 'dy'.
    allow_none : bool (optional - default=False)
        If True, return None if not found.

    Returns
    -------
        vard: xarray.core.dataarray.DataArray
            Output data field.

    @author: Peter Clark

    """
    logger.info(f'Retrieving {var_name:s}.')
    
    try:
        if var_name in source_dataset or f'STASH_{var_name}' in source_dataset:
            vard = get_um_field(source_dataset, 
                                name=var_name, 
                                add_name=add_name)
        elif var_name in stash_map:
            vard = get_um_field(source_dataset, stash=stash_map[var_name])
            vard.name = var_name           

        elif options is not None \
            and 'aliases' in options \
            and var_name in options.get('aliases',[]):
                
            for alias in options['aliases'][var_name]:
                if alias in source_dataset:
                    logger.info(f'Retrieving {alias:s} as {var_name:s}.')
                    vard = get_um_field(source_dataset, 
                                        name=alias)
                    vard.name = var_name
                    break
                elif alias in stash_map:
                    logger.info(f'Retrieving {alias:s} as {stash_map[var_name]:s}.')
                    vard = get_um_field(source_dataset, 
                                        stash=stash_map[var_name])
                    vard.name = var_name           
                    break
            else:
                raise KeyError(f"Cannot retrieve {var_name}")
        else:
            raise KeyError(f"Cannot retrieve {var_name}")

        # Change 'timeseries...' variable to 'time'
        # [itime] = get_string_index(vard.dims, ['time'])
        # if itime is not None:
            # vard = vard.rename({vard.dims[itime]: 'time'})

    except KeyError:
               
        vard = get_um_processed_var(source_dataset,
                                    var_name,
                                    options=options,
                                    allow_none=allow_none)
        
    if vard is None :
        if allow_none:
            return None
        else:
            raise KeyError(f"Data {var_name:s} not in dataset.")

    # else:    

        # vard = correct_um_grid_and_units(var_name, vard, source_dataset,
                                         # options=options)
    

    return vard

def correct_um_grid_and_units(var_name: str,
                              vard: xr.core.dataarray.DataArray,
                              source_dataset: xr.core.dataset.Dataset,
                              options: dict=None):
    """
    Correct input grid specification.
    SPECIFIC TO UM

    Parameters
    ----------
    var_name : str
        Name of variable to retrieve.
    vard : xr.core.dataarray.DataArray
        Input (at least 2D) data.
    source_dataset : xr.core.dataset.Dataset
        Source dataset for vard
    options : dict(optional - default=None)
        Options possibly used are 'dx' and 'dy'.

    Returns
    -------
    vard : xarray
        Required data with corrected grid.

    """
    
    return vard
    # Get model resolution values
    if 'dx' not in vard.attrs or 'dx' not in vard.attrs:   
        raise ValueError("Grid info not available")
    else:
        dx = vard.attrs['dx']
        dy = vard.attrs['dy']

    # Add correct x and y grids.

    if var_name in var_properties:

        vp = var_properties[var_name]['grid']

        if 'x' in vard.dims:
            nx = vard.shape[vard.get_axis_num('x')]

            if vp[0] :
                x = (np.arange(nx) + 1.0) * np.float64(dx)
                xn = 'x_u'
            else:
                x = (np.arange(nx) + 0.5) * np.float64(dx)
                xn = 'x_p'

            vard = vard.rename({'x':xn})
            vard.coords[xn] = x

        if 'y' in vard.dims:
            ny = vard.shape[vard.get_axis_num('y')]
            if vp[1] :
                y = (np.arange(ny) + 1.0) * np.float64(dy)
                yn = 'y_v'
            else:
                y = (np.arange(ny) + 0.5)* np.float64(dy)
                yn = 'y_p'

            vard = vard.rename({'y':yn})
            vard.coords[yn] = y

        if 'z' in vard.dims:
            if vp[2]:
                vard = vard.rename({'z':'z_w'})
            else:
                zn = source_dataset.coords['zn']
                vard = vard.rename({'z':'z_p'})
                vard.coords['z'] = zn.data

        if 'zn' in vard.dims:
            if vp[2]:
                z = source_dataset.coords['z']
                vard = vard.rename({'zn':'z_w'})
                vard.coords['z_w'] = z.data
            else:
                vard = vard.rename({'zn':'z_p'})


        vard.attrs['units'] = var_properties[var_name]['units']

    else:

        if 'x' in vard.dims:
            nx = vard.shape[vard.get_axis_num('x')]
            x = (np.arange(nx) + 0.5) * np.float64(dx)
            xn = 'x_p'
            vard = vard.rename({'x':xn})
            vard.coords[xn] = x

        if 'y' in vard.dims:
            ny = vard.shape[vard.get_axis_num('y')]
            y = (np.arange(ny) + 0.5) * np.float64(dy)
            yn = 'y_p'
            vard = vard.rename({'y':yn})
            vard.coords[yn] = y

        if 'z' in vard.dims:
            vard = vard.rename({'z':'z_w'})

        if 'zn' in vard.dims:
            vard = vard.rename({'zn':'z_p'})

        if 'units' not in vard.attrs:        
            vard.attrs['units'] = ''
            
    vard = set_grid_type(vard, grid_type_def='xy_periodic') 
    vard = set_coord_type(vard, coord_type_def='cartesian')              

    for c in 'xy':        
        field = add_grid_attrs(vard, c, xy_periodic=True)

    return vard


def get_um_processed_var(source_dataset,
                         var_name: str,
                         options: dict=None,
                         allow_none: bool=False):
    if var_name == 'piref':
        vard = get_derived_vars(source_dataset,
                                'exner', th.derived_vars,
                                options=options,
                                get_data_fn=get_um_data)
        vard = get_mean(vard)
        vard.name = var_name  
        
    elif var_name[-3:] == 'ref':
        vard = get_um_data(source_dataset, var_name[:-3])
        vard = get_mean(vard)
        vard.name = var_name  
                                
    elif var_name in th.derived_vars:

        vard = get_derived_vars(source_dataset, 
                                   var_name, 
                                   th.derived_vars,
                                   options=options,
                                   get_data_fn=get_um_data)
    elif var_name[:4] == 'dbyd':
                
        vard = get_derivative(source_dataset,
                              var_name,
                              options=options,
                              allow_none=allow_none,
                              get_data_fn=get_um_data) 
            

    else :
        
        vard = None
        
    return vard

def get_um_and_transform(source_dataset, var_name,
                         options=None,
                         add_name=True,
                         grid='p'):
    """
    Extract data from dataset and transform to alternative grid.

    See get_data for derived variables.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    var_name : str
        Name of variable to retrieve.
    options : dict (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.
    grid : str, optional
        Destination grid 'u', 'v', 'w' or 'p'. Default is 'p'.

    Returns
    -------
        var: xarray.core.dataarray.DataArray
            Output data field.

    @author: Peter Clark

    """
    def _get_mapped_coord(source_dataset, c_aux, c_main, c_new):
        if c_aux in source_dataset.coords:
            c = source_dataset.coords[c_aux].rename({c_aux:c_new})
            c = c.swap_dims({c_main:c_new})
        elif c_main in source_dataset.dims:
            c = source_dataset.coords[c_main]
        else:
            raise KeyError(f"Cannot find {c} in dataset.")
        c.name = c_new
        return clean_coords(c)
        
    
    var = get_um_data(source_dataset, var_name, 
                      options=options, 
                      add_name=add_name)
    
    # print(f'{var=}')
    
    if 'z_w' in var.dims:
        z_w = var.z_w
        z_p_values = 0.5 * (z_w.values[0:-1] + z_w.values[1:])
        z_p = xr.DataArray(z_p_values, {'z_p':z_p_values}, name='z_p')
        
    elif 'z_p' in var.dims:
        z_p = var.z_p
        z_w_values = 0.5 * (z_p.values[0:-1] + z_p.values[1:])
        z_w_top = z_w_values[-1] * 2 - z_w_values[-2]
        z_w_values = np.append(z_w_values, z_w_top)
        z_w = xr.DataArray(z_w_values, {'z_w':z_w_values}, name='z_w')
        
    else:
                       
        if 'thlev_eta_theta'  in source_dataset.dims and \
           'thlev_zsea_theta' in source_dataset.coords:
            
            z_w = _get_mapped_coord(source_dataset, 
                                    'thlev_zsea_theta', 
                                    'thlev_eta_theta',
                                    'z_w')
        elif 'level_height_w' in source_dataset.dims:
            
            z_w = source_dataset.coords['level_height_w'].rename(
                                       {'level_height_w':'z_w'})
    
        if 'rholev_eta_rho'  in source_dataset.dims and \
             'rholev_zsea_rho' in source_dataset.coords:
        
            z_p = _get_mapped_coord(source_dataset,
                                    'rholev_zsea_rho',
                                    'rholev_eta_rho',
                                    'z_p')
        elif 'level_height_p' in source_dataset.dims:
            
            z_p = source_dataset.coords['level_height_p'].rename(
                                       {'level_height_p':'z_p'})

        
    var = do.grid_conform(var, z_w, z_p, grid = grid )

    # Re-chunk data if using dask
    if not monc_utils.global_config['no_dask']:
        var = re_chunk(var)
#    logger.info(var)

    return var

def get_um_data_on_grid(source_dataset, var_name,
                        derived_dataset=None,
                        options=None,
                        rename_time=False,
                        add_name=True,
                        grid='p') :
    """
    Find data from source_dataset remapped to destination grid.

    Uses data from derived_dataset if present, otherwise uses
    get_and_transform to input from source_dataset and remap grid.
    In this case, if options['save_all']=='yes', save the remapped data to
    derived_dataset.

    See get_data for derived variables.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    var_name : str
        Name of variable to retrieve.
    derived_dataset : dict, optional
        'ds' points to xarray Dataset, 'file' to output file path.
    options : dict, (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.
    grid : str, optional
        Destination grid 'u', 'v', 'w' or 'p'. Default is 'p'.

    Returns
        var: xarray.core.dataarray.DataArray
            Output data field.

    @author: Peter Clark
    """
    ongrid = '_on_'+grid

    # Logic here:
    # If var_name already qualified with '_on_x', where x is a grid
    # then if x matches required output grid, see if in derived_dataset
    # already, and use if it is.
    # Otherwise strip '_on_x' and go back to source data as per default.

    # First, find op_var_name
    # Default
    
    if re.match(stash_pattern, var_name) and add_name:
        field_name = inverse_lookup(var_name, stash_map)
        op_var_name = field_name + ongrid
    else:    
        op_var_name = var_name + ongrid
        if len(var_name) > 5:
            if var_name[-5:] == ongrid:
                op_var_name = var_name
            elif var_name[-5:-1] == '_on_':
                var_name = var_name[:-5]
                op_var_name = var_name[:-5] + ongrid

    if options is not None and options.get('save_all', 'yes').lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name in derived_dataset['ds'].variables:

            op_var = derived_dataset['ds'][op_var_name]
            logger.info(f'Retrieved {op_var_name:s} from derived dataset.')
            return op_var

    op_var = get_um_and_transform(source_dataset,
                                  var_name, 
                                  options=options, 
                                  add_name=add_name,
                                  grid=grid)
    op_var.name = op_var_name
    
    if rename_time:
        [itime] = get_string_index(op_var.dims, ['time'])
        if itime is not None:
            op_var = op_var.rename({op_var.dims[itime]:'time'})
            
    if options is not None and options.get('save_all', 'yes').lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name not in derived_dataset['ds'].variables:
            op_var = save_field(derived_dataset, op_var)

    return op_var

def get_mean(vard):
    if um_datain_options['ref_is_init_mean']:
        [itime] = get_string_index(vard.dims, ['time'])
        if itime is not None:
            tvar = vard.dims[itime]
            vard = vard.isel({tvar:0})
        
    [ix, iy] = get_string_index(vard.dims, ['x', 'y'])
    xvar = vard.dims[ix]
    yvar = vard.dims[iy]
    
    vard = vard.mean(dim=(xvar, yvar))
    return vard

def get_um_coords(field):
    clon = [c for c in field.coords if 'longitude' in c][0]
    clat = [c for c in field.coords if 'latitude' in c][0]    
    cz = [c for c in field.coords if 'z' in c]
    if len(cz) > 0 : 
        cz = cz[0] 
    else: 
        cz = None 
    ct = [c for c in field.coords if 'min' in c]
    if len(ct) > 0:
        ct = ct[0]
    else:
        ct = [c for c in field.coords if 'time' in c]
        if len(ct) > 0:
            ct = ct[0]
        else:
            ct = None
    return (clon, clat, cz, ct)

def get_um_grid_desc(cg):
    gd = cg.split('_')
    if len(gd) <= 1: return ''
    if gd[1][0] == 't': return '_p'
    if gd[1][0] == 'c': 
        grid_letter = gd[1][1]        
        return f'_{grid_letter}'
    else: return f'_{gd[1]}'
