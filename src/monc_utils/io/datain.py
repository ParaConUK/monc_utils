"""
Created on Mon Aug  2 11:01:11 2021.

@author: Peter Clark
"""
import numpy as np
import xarray as xr
from monc_utils.io.file_utils import (options_database,
                                      configure_model_resolution,
                                      )
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.dataout import save_field
from monc_utils.data_utils.dask_utils import re_chunk

import monc_utils.data_utils.difference_ops as do
import monc_utils.thermodynamics.thermodynamics as th
import monc_utils.thermodynamics.thermodynamics_constants as thc
import monc_utils
import re
from loguru import logger

PGRID = [False,False,False]
UGRID = [True,False,False]
VGRID = [False,True,False]
WGRID = [False,False,True]
TGRID = PGRID
BGRID = [True,True,False]

"""
Hierarchy 

get_data_on_grid
    get_and_transform
        get_data
            get_data
            get_thref
            get_pref
            exner
            get_derived_vars
                get_data
            get_derivative
                get_data
                d_by_dx_field_native
                d_by_dy_field_native
                d_by_dz_field_native
            correct_grid_and_units
        grid_conform
    save_field
    
    

"""

def is_xy_periodic(field:xr.DataArray, 
                   xy_periodic_def:bool=True) -> (xr.DataArray, bool):
    """
    Find if field is xy_periodic and set attribute in field.
    Returns field.attrs['xy_periodic'] if available, otherwise default.

    Parameters
    ----------
    field : xr.DataArray
        Input MONC or UM field.
    xy_periodic_def : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    field : xr.DataArray 
        Input field with attrs['xy_periodic'] set..
    xy_periodic : bool
        True if xy_periodic.

    """
    
    if 'xy_periodic' not in field.attrs:
        xy_periodic = xy_periodic_def
        field.attrs['xy_periodic'] = xy_periodic
    else:
        xy_periodic = field.attrs['xy_periodic']
        
    return field, xy_periodic

def get_full_dim_name(field:xr.DataArray, dimname:str) -> str:
    """
    Find dimension name in input field that contains dimname.

    Parameters
    ----------
    field : xr.DataArray
        Any xr.DataArray
    dimname : str
        Part of possible dimname. Typical use is dimname='longitude' to match 
        both 'longitude' and 'grid_longitude'.

    Returns
    -------
    str or None
        Corresponding dim if present.

    """
    
    full_dim = [d for d in field.dims if dimname in d]
    if full_dim : 
        full_dim = full_dim[0]
    else:
        full_dim = None
        
    return full_dim

def add_grid_attrs(field:xr.DataArray, coord_name:str, 
                   xy_periodic:bool=True) -> xr.DataArray:
    """
    Add 'xy_periodic', drid spacing and domain size to field.attrs.

    Parameters
    ----------
    field : xr.DataArray
        Any xr.DataArray.
    coord_name : str
        dimension name or name part.
    xy_periodic : bool, optional
        xy_periodic or not. The default is True.

    Returns
    -------
    field : xr.DataArray
        Input field with modified attrs.

    """
    
    if 'xy_periodic' in field.attrs:
        xy_periodic = field.attrs['xy_periodic']
    else:
        if xy_periodic is None:
            xy_periodic = True
        field.attrs['xy_periodic'] = xy_periodic
        
    coord_name_full = get_full_dim_name(field, coord_name)
    
    if coord_name_full is None: return field

    coord = field[coord_name_full]
    
    delta = coord.diff(dim=coord_name_full).min().item()
    
    if xy_periodic:
        domain = coord.values[-1] - coord.values[0] + delta
    else:
        domain = coord.values[-1] - coord.values[0]
    
    field.attrs[f'd{coord_name[0]}'] = delta
    field.attrs[f'L{coord_name[0]}'] = domain
    
    return field
    
def clean_coords(field:xr.DataArray, keep_coords=None) -> xr.DataArray:
    """
    Remove unwanted non-dimensional coords from field.
    Will not remove 'elapsed_time'

    Parameters
    ----------
    field : xr.DataArray
        Any xr.DataArray.
    keep_coords : list or None, optional
        List of coords to keep. The default is None.

    Returns
    -------
    field : xr.DataArray
        Input DataArray cleaned of unwanted coords.

    """
    
    if keep_coords is None: keep_coords = []

    for c in field.coords:
        if(c in field.dims 
           or c == 'elapsed_time' 
           or any([d in c for d in keep_coords]) ): continue
        field = field.drop_vars(c)
        
    return field

def correct_grid_and_units(var_name: str,
                           vard: xr.core.dataarray.DataArray,
                           source_dataset: xr.core.dataset.Dataset,
                           options: dict=None):
    """
    Correct input grid specification.
    SPECIFIC TO MONC

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
    #   Mapping of data locations on grid via logical triplet:
    #   logical[u-point,v-point,w-point]
    #          [False,  False,  False  ] --> (p,th,q)-point
    var_properties = {"u":{"grid":UGRID,
                           "units":'m.s-1'},
                      "v":{"grid":VGRID,
                           "units":'m.s-1'},
                      "w":{"grid":WGRID,
                           "units":'m.s-1'},
                      "th":{"grid":TGRID,
                            "units":'K'},
                      "theta":{"grid":TGRID,
                            "units":'K'},
                      "thref":{"grid":TGRID,
                            "units":'K'},
                      "p":{"grid":PGRID,
                           "units":'Pa'},
                      "pressure":{"grid":PGRID,
                           "units":'Pa'},
                      "pref":{"grid":PGRID,
                           "units":'Pa'},
                      "q_vapour":{"grid":TGRID,
                                  "units":'kg/kg'},
                      "q_cloud_liquid_mass":{"grid":TGRID,
                                             "units":'kg/kg'},
                      "q_ice_mass":{"grid":TGRID,
                                    "units":'kg/kg'},
                      "u_b":{"grid":BGRID,
                                             "units":'m.s-1'},
                      "v_b":{"grid":BGRID,
                                             "units":'m.s-1'},
                      }

    # Get model resolution values
    if 'dx' not in vard.attrs or 'dx' not in vard.attrs:   
        dx, dy, options = configure_model_resolution(source_dataset,
                                                     options=options)
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
            
    for c in 'xy':        
        field = add_grid_attrs(vard, c, xy_periodic=True)

    return vard

def get_data(source_dataset, 
             var_name: str,
             options: dict=None,
             allow_none: bool=False) :
    """
    Extract data or derived data field from source NetCDF dataset.

    If var_name is in source_dataset it is retrieved; if one of the primary
    variables with a key in var_properties the grid is corrected.
    Otherwise, it is assumed to be on a 'p' point.

    Currently written for MONC data, enforcing C-grid. Returned coords are
    'x_p', 'x_u', 'y_p', 'y_v', 'z_w', 'z_p'. Coordinate x- and -y values are
    retrieved from the MONC options_database in source_dataset
    or from 'dx' and 'dy' in options otherwise. 

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
        vard: xarray
            Output data field.

    @author: Peter Clark

    """
    
    
    logger.info(f'Retrieving {var_name:s}.')
    
    vard = None
    
    try:
        if var_name in source_dataset:
            vard = source_dataset[var_name]

        elif options is not None \
            and 'aliases' in options \
            and var_name in options.get('aliases',[]):
            for alias in options['aliases'][var_name]:
                if alias in source_dataset:
                    logger.info(f'Retrieving {alias:s} as {var_name:s}.')
                    vard = source_dataset[alias]
                    vard.name = var_name
                    break
            else:
                raise KeyError(f"Cannot retrieve {var_name}")
        else:
            raise KeyError(f"Cannot retrieve {var_name}")

        # Change 'timeseries...' variable to 'time'
        [itime] = get_string_index(vard.dims, ['time'])
        if itime is not None:
            if vard.dims[itime] != 'time':
                vard = vard.rename({vard.dims[itime]: 'time'})

    except KeyError:
        
        match var_name:
            case 'theta' :
                thp = get_data(source_dataset, 'th', 
                               options=options, allow_none=allow_none) 
                thref = get_data(source_dataset, 'thref', 
                               options=options, allow_none=allow_none)
                vard = thp + thref
                vard.name = 'theta'
            case 'pressure' :
                p = get_data(source_dataset, 'p', 
                             options=options, allow_none=allow_none) 
                pref = get_data(source_dataset, 'pref', 
                             options=options, allow_none=allow_none)
                vard = p + pref
                vard.name = 'pressure'
            case 'thref':
                vard = get_thref(source_dataset, options=options)
            case 'pref':
                vard = get_pref(source_dataset, options=options)
            case 'piref':
                vard = th.exner(get_pref(source_dataset, options=options))
            case 'z':
                vard = source_dataset.dims['z']
            case 'zn':
                vard = source_dataset.dims['zn']
            case _:
                
                if var_name in th.derived_vars:

                    vard = get_derived_vars(source_dataset,
                                            var_name, th.derived_vars,
                                            options=options)
            
                elif var_name[:4] == 'dbyd':
                    
                    vard = get_derivative(source_dataset,
                                         var_name,
                                         options=options,
                                         allow_none=allow_none) 
                    
                else:
                    
                    vard = None
                        

    if vard is None :
        if allow_none:
            return None
        else:
            raise KeyError(f"Data {var_name:s} not in dataset.")

    else:                    
        
        vard = correct_grid_and_units(var_name, vard, source_dataset,
                                      options=options)

    return vard


def get_derived_vars(source_dataset,
                     var_name: str, derived_vars: dict, 
                     options: dict=None,
                     get_data_fn: callable=get_data):
    """
    Get data from source_dataset and compute required variable.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    var_name : str
        Name of variable to retrieve.
    derived_vars : dict
        Maps var_name to function name and argument list.
    options : dict (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.
    get_data_fn: callable
        Function used to 

    Returns
    -------
    vard : TYPE
        DESCRIPTION.

    """
    dv = derived_vars[var_name]
    args = []
    for v in dv['vars']:
        allow_none=False
        if v[0] == '[':
            allow_none=True
            v = v[1:-1]
        var = get_data_fn(source_dataset, v, 
                       options=options,
                       allow_none=allow_none)
        if var is not None:
            args.append(var)
        else:
            logger.info(f'{v} not in dataset.')
    vard = dv['func'](*args)
    vard.name = var_name
    vard.attrs['units'] = dv['units']
    return vard

def get_derivative(source_dataset,
                   var_name: str,
                   options: dict=None,
                   allow_none: bool=False,
                   get_data_fn: callable=get_data) :
    """
    Get data from source_dataset and compute required variable.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    var_name : str
        Should be of form dbyds(variable) where variable is name of 
        variable to retrieve and s=x or y or z. 
    options : dict (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.
    allow_none : bool (optional - default=False)
        If True, return None if not found.

    Returns
    -------
    vard : TYPE
        DESCRIPTION.

    """


    deriv = re.compile(r'dbyd[xyz]\(*')
    mo = deriv.match(var_name)
    
    if mo is None:
        return None

    target_var_name = var_name[mo.end():]
    if target_var_name[-1] != ')':
        raise KeyError(f"Data {var_name:s} not in dataset.")
        
    target_var_name = target_var_name[:-1]
    
    target_var = get_data_fn(source_dataset, 
                             target_var_name,
                             options=options,
                             allow_none=allow_none)
    
    # No else required as match guaranteed above.
    if var_name[4] == 'x':
        vard = do.d_by_dx_field_native(target_var)
    elif var_name[4] == 'y':
        vard = do.d_by_dy_field_native(target_var)
    elif var_name[4] == 'z':                    
        vard = do.d_by_dz_field_native(target_var )
        
    # The following should be a null operation for derivatives.
    
    vard.attrs.update(target_var.attrs)

    vard = correct_grid_and_units(var_name, 
                                  vard, 
                                  source_dataset,
                                  options=options)
    return vard

    

def get_and_transform(source_dataset, var_name,
                      options=None,
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
        var: xarray
            Output data field.

    @author: Peter Clark

    """
    var = get_data(source_dataset, var_name, options=options)
    if "z" in source_dataset.dims:
        z_w = source_dataset["z"].rename({'z':'z_w'})
    elif "z_w" in source_dataset.dims:
        z_w = source_dataset["z_w"]
    else:
        raise KeyError("Cannot find z in dataset.")
    if "zn" in source_dataset.dims:
        z_p = source_dataset["zn"].rename({'zn':'z_p'})
    elif "z_p" in source_dataset.dims:
        z_p = source_dataset["z_p"]
    else:
        raise KeyError("Cannot find zn in dataset.")

    var = do.grid_conform(var, z_w, z_p, grid = grid )

    # Re-chunk data if using dask
    if not monc_utils.global_config['no_dask']:
        var = re_chunk(var)
#    logger.info(var)

    return var

def get_data_on_grid(source_dataset, var_name,
                     derived_dataset=None,
                     options=None,
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
    -------
        var: xarray
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
    op_var_name = var_name + ongrid

    if len(var_name) > 5:
        if var_name[-5:] == ongrid:
            op_var_name = var_name
        elif var_name[-5:-1] == '_on_':
            var_name = var_name[:-5]
            op_var_name = var_name[:-5] + ongrid

    if options is not None and options.get('save_all','yes').lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name in derived_dataset['ds'].variables:

            op_var = derived_dataset['ds'][op_var_name]
            logger.info(f'Retrieved {op_var_name:s} from derived dataset.')
            return op_var

    op_var = get_and_transform(source_dataset, 
                               var_name, options=options, grid=grid)
    op_var.name = op_var_name

    if options is not None and options.get('save_all', 'yes').lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name not in derived_dataset['ds'].variables:
            op_var = save_field(derived_dataset, op_var)

    return op_var

def get_pref(source_dataset, options=None):
    """
    Get reference pressure profile for source_dataset.

    Calculate from ref_dataset or from surface_press in source_dataset
    options_database and options['th_ref'].

    Parameters
    ----------
    source_dataset :  netCDF4 file
        MONC output file.
    ref_dataset :  netCDF4 file or None
        MONC output file containing 1D variable prefn.
    options : dict (optional - default=None)
        Options. Options possibly used are th_ref.

    Returns
    -------
    pref

    """
         
    if 'prefn'  not in source_dataset.data_vars:
        od = options_database(source_dataset)
        if od is not None:
            p_surf = float(od['surface_pressure'])
        else:
            p_surf = thc.p_ref_theta

        if options is None:
            thref = 300.0
        else:
            thref = options['th_ref']

        zn = source_dataset['zn']
        piref0 = (p_surf/thc.p_ref_theta)**thc.kappa
        piref = piref0 - (thc.g/(thc.cp_air * thref)) * zn
        pref = thc.p_ref_theta * piref**thc.rk
    else:
        pref = source_dataset['prefn']
        [itime] = get_string_index(pref.dims, ['time'])
        if itime is not None:
            tvar = list(pref.dims[itime])
            pref = pref.isel({tvar:0}).squeeze(drop=True).drop(tvar)
        pref = correct_grid_and_units('pref', pref, source_dataset,
                                      options=options)
            
    pref.attrs['units'] = 'Pa'

    return pref

def get_thref(source_dataset, options=None):
    """
    Get thref profile from ref_dataset.

    Parameters
    ----------
    source_dataset : netCDF4 file or None
        MONC output file containing pref
    options : dict
        Options. Options possibly used are th_ref.

    Returns
    -------
    thref : float or float array.
        Reference theta constant or profile

    """

    if source_dataset is None or 'thref'  not in source_dataset.data_vars:
        if options is None:
            thref = 300.0
        else:
            thref = options['th_ref']
        thref = xr.DataArray(thref, dims=['time'], coords={'time':[0.0]})
        
    else:
        thref = source_dataset['thref']
        [itime] = get_string_index(thref.dims, ['time'])
        if itime is not None:
            tvar = thref.dims[itime]
            thref = thref.isel({tvar:0}).squeeze(drop=True).drop(tvar)
        thref = correct_grid_and_units('thref', thref, source_dataset,
                                        options=options)
            
    thref.attrs['units'] = 'K'

    return thref
