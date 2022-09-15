"""
Created on Mon Aug  2 11:01:11 2021.

@author: Peter Clark
"""
import numpy as np
import xarray
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

def correct_grid_and_units(var_name: str,
                           vard: xarray.core.dataarray.DataArray,
                           source_dataset: xarray.core.dataset.Dataset,
                           options: dict=None):
    """
    Correct input grid specification.

    Parameters
    ----------
    var_name : str
        Name of variable to retrieve.
    vard : xarray.core.dataarray.DataArray
        Input (at least 2D) data.
    source_dataset : xarray.core.dataset.Dataset
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
    var_properties = {"u":{"grid":[True,False,False],
                           "units":'m.s-1'},
                      "v":{"grid":[False,True,False],
                           "units":'m.s-1'},
                      "w":{"grid":[False,False,True],
                           "units":'m.s-1'},
                      "th":{"grid":[False,False,False],
                            "units":'K'},
                      "p":{"grid":[False,False,False],
                           "units":'Pa'},
                      "q_vapour":{"grid":[False,False,False],
                                  "units":'kg/kg'},
                      "q_cloud_liquid_mass":{"grid":[False,False,False],
                                             "units":'kg/kg'},
                      "q_ice_mass":{"grid":[False,False,False],
                                    "units":'kg/kg'},
                      }

    # Get model resolution values
    dx, dy, options = configure_model_resolution(source_dataset,
                                                 options=options)

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

    return vard

def get_derived_vars(source_dataset, ref_dataset,
                     var_name: str, derived_vars: dict, options: dict=None):
    """
    Get data from source_dataset and compute required variable.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    ref_dataset : xarray Dataset
        Contains reference profiles. Can be None.
    var_name : str
        Name of variable to retrieve.
    derived_vars : dict
        Maps var_name to function name and argument list.
    options : dict (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.

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
        var = get_data(source_dataset, ref_dataset, v, options=options,
                       allow_none=allow_none)
        if var is not None:
            args.append(var)
        else:
            print(f'{v} not in dataset.')
    vard = dv['func'](*args)
    vard.name = var_name
    vard.attrs['units'] = dv['units']
    return vard
    
def get_data(source_dataset, ref_dataset, var_name: str,
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
    ref_dataset :  xarray Dataset
        Contains reference profiles. Can be None.
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
    print(f'Retrieving {var_name:s}.')
    try:
        if var_name in source_dataset:
            vard = source_dataset[var_name]

        elif options is not None \
            and 'aliases' in options \
            and var_name in options['aliases']:
            for alias in options['aliases'][var_name]:
                if alias in source_dataset:
                    print(f'Retrieving {alias:s} as {var_name:s}.')
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
            vard = vard.rename({vard.dims[itime]: 'time'})

        if var_name == 'th' :
            thref = get_thref(ref_dataset,
                              options=options)
            vard += thref

        if var_name == 'p' :
            pref = get_pref(source_dataset, ref_dataset,
                            options=options)
            vard += pref


    except KeyError:
               
        if var_name == 'thref' :
            vard = get_thref(ref_dataset, options=options)
        elif var_name == 'pref' :
            vard = get_pref(source_dataset, ref_dataset, options=options)
        elif var_name == 'piref' :
            vard = th.exner(get_pref(source_dataset, ref_dataset,
                                     options=options))
        elif var_name == 'z' :
            vard = ref_dataset.dims['z']
        elif var_name == 'zn' :
            vard = ref_dataset.dims['zn']

        elif var_name in th.derived_vars:

            vard = get_derived_vars(source_dataset, ref_dataset,
                                    var_name, th.derived_vars,
                                    options=options)
            
        else:
            
            deriv = re.compile(r'dbyd[xyz]\(*')
            mo = deriv.match(var_name)
            
            if mo is not None:
        
                target_var_name = var_name[mo.end():]
                if target_var_name[-1] != ')':
                    raise KeyError(f"Data {var_name:s} not in dataset.")
                    
                target_var_name = target_var_name[:-1]
                
                target_var = get_data(source_dataset, 
                                      ref_dataset, 
                                      target_var_name,
                                      options=options,
                                      allow_none=allow_none)
                if var_name[4] == 'x':
                    vard = do.d_by_dx_field_native(target_var)
                elif var_name[4] == 'y':
                    vard = do.d_by_dy_field_native(target_var)
                elif var_name[4] == 'z':                    
                    vard = do.d_by_dz_field_native(target_var )
                    vard = correct_grid_and_units(var_name, 
                                                  vard, 
                                                  source_dataset,
                                                  options=options)
                    
                return vard
                

            else :
                if allow_none:
                    return None
                else:
                    raise KeyError(f"Data {var_name:s} not in dataset.")
                
    vard = correct_grid_and_units(var_name, vard, source_dataset,
                                  options=options)

    return vard

def get_and_transform(source_dataset, ref_dataset, var_name,
                      options=None,
                      grid='p'):
    """
    Extract data from dataset and transform to alternative grid.

    See get_data for derived variables.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    ref_dataset : xarray Dataset
        Contains reference profiles. Can be None.
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
    var = get_data(source_dataset, ref_dataset, var_name, options=options)
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
#    print(var)

    return var

def get_data_on_grid(source_dataset, ref_dataset, var_name,
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
    ref_dataset : xarray Dataset
        Contains reference profiles. Can be None.
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

    # First, find op_name
    # Default
    op_var_name = var_name + ongrid

    if len(var_name) > 5:
        if var_name[-5:] == ongrid:
            op_var_name = var_name
        elif var_name[-5:-1] == '_on_':
            var_name = var_name[:-5]
            op_var_name = var_name[:-5] + ongrid

    if options is not None and options['save_all'].lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name in derived_dataset['ds'].variables:

            op_var = derived_dataset['ds'][op_var_name]
            print(f'Retrieved {op_var_name:s} from derived dataset.')
            return op_var

    op_var = get_and_transform(source_dataset, ref_dataset,
                               var_name, options=options, grid=grid)
    op_var.name = op_var_name

    if options is not None and options['save_all'].lower() == 'yes':

        if derived_dataset is not None \
            and op_var_name not in derived_dataset['ds'].variables:
            op_var = save_field(derived_dataset, op_var)

    return op_var

def get_pref(source_dataset, ref_dataset,  options=None):
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
    (pref, piref)

    """
    if ref_dataset is None:
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
#        pref = xarray.DataArray(pref, dims=['time'], coords={'time':[0.0]})

#                print('pref', pref)
    else:
        pref = ref_dataset['prefn']
        [itime] = get_string_index(pref.dims, ['time'])
        if itime is not None:
            tvar = pref.dims[itime]
            pref = pref.isel({tvar:0}).squeeze(drop=True).drop(tvar)
            # tdim = pref.dims[itime]
            # pref = pref[{tdim:[0]}].squeeze()
            # pref = pref.drop_vars(tdim)

    pref.attrs['units'] = 'Pa'

    return pref

def get_thref(ref_dataset, options=None):
    """
    Get thref profile from ref_dataset.

    Parameters
    ----------
    ref_dataset : netCDF4 file or None
        MONC output file containing pref
    options : dict
        Options. Options possibly used are th_ref.

    Returns
    -------
    thref : float or float array.
        Reference theta constant or profile

    """
    if ref_dataset is None:
        if options is None:
            thref = 300.0
        else:
            thref = options['th_ref']
        thref = xarray.DataArray(thref, dims=['time'], coords={'time':[0.0]})
        
    else:
        thref = ref_dataset['thref']
        [itime] = get_string_index(thref.dims, ['time'])
        if itime is not None:
            tvar = thref.dims[itime]
            thref = thref.isel({tvar:0}).squeeze(drop=True).drop(tvar)
            
    thref.attrs['units'] = 'K'

            # tdim = thref.dims[itime]
            # thref = thref[{tdim:[0]}].squeeze()
            # thref = thref.drop_vars(tdim)

    return thref
