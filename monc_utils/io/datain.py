"""
Created on Mon Aug  2 11:01:11 2021.

@author: Peter Clark
"""
import sys
import numpy as np
import re
import xarray
from monc_utils.io.file_utils import options_database
from monc_utils.data_utils.string_utils import get_string_index
#from monc_utils.utils.dask_utils import re_chunk
from monc_utils.io.dataout import save_field
import monc_utils.data_utils.difference_ops as do
import monc_utils.thermodynamics.thermodynamics as th
import monc_utils.thermodynamics.thermodynamics_constants as thc
import monc_utils

def correct_grid_and_units(var_name: str,
                           vard: xarray.core.dataarray.DataArray,
                           source_dataset: xarray.core.dataset.Dataset,
                           options: dict):
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
    options : dict
        Options possibly used are 'dx' and 'dy'.

    Returns
    -------
    vard : TYPE
        DESCRIPTION.

    """
    #   Mapping of data locations on grid via logical triplet:
    #   logical[u-point,v-point,w-point]
    #          [False,  False,  False  ] --> (p,th,q)-point
    var_properties = {"u":{"grid":[True,False,False],
                           "units":'m s-1'},
                      "v":{"grid":[False,True,False],
                           "units":'m s-1'},
                      "w":{"grid":[False,False,True],
                           "units":'m s-1'},
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
    dx, dy, options = configure_model_resolution(source_dataset, options)

    # Add correct x and y grids.

    if var_name in var_properties:

        vp = var_properties[var_name]['grid']

        if 'x' in vard.dims:
            nx = vard.shape[vard.get_axis_num('x')]

            if vp[0] :
                x = (np.arange(nx) + 0.5) * np.float64(dx)
                xn = 'x_u'
            else:
                x = np.arange(nx) * np.float64(dx)
                xn = 'x_p'

            vard = vard.rename({'x':xn})
            vard.coords[xn] = x

        if 'y' in vard.dims:
            ny = vard.shape[vard.get_axis_num('y')]
            if vp[1] :
                y = (np.arange(ny) + 0.5) * np.float64(dy)
                yn = 'y_v'
            else:
                y = np.arange(ny) * np.float64(dy)
                yn = 'y_p'

            vard = vard.rename({'y':yn})
            vard.coords[yn] = y

        if 'z' in vard.dims and not vp[2]:
            zn = source_dataset.coords['zn']
            vard = vard.rename({'z':'zn'})
            vard.coords['zn'] = zn.data

        if 'zn' in vard.dims and vp[2]:
            z = source_dataset.coords['z']
            vard = vard.rename({'zn':'z'})
            vard.coords['z'] = z.data

        vard.attrs['units'] = var_properties[var_name]['units']

    else:

        if 'x' in vard.dims:
            nx = vard.shape[vard.get_axis_num('x')]
            x = np.arange(nx) * np.float64(dx)
            xn = 'x_p'
            vard = vard.rename({'x':xn})
            vard.coords[xn] = x

        if 'y' in vard.dims:
            ny = vard.shape[vard.get_axis_num('y')]
            y = np.arange(ny) * np.float64(dy)
            yn = 'y_p'
            vard = vard.rename({'y':yn})
            vard.coords[yn] = y

        vard.attrs['units'] = ''

    return vard

def get_derived_vars(source_dataset, ref_dataset, options,
                     var_name: str, derived_vars: dict):
    """
    Get data from source_dataset and compute required variable.

    Parameters
    ----------
    source_dataset : xarray Dataset
        Input (at least 2D) data.
    ref_dataset : xarray Dataset
        Contains reference profiles. Can be None.
    options : dict
        Options. Options possibly used are 'dx' and 'dy'.
    var_name : str
        Name of variable to retrieve.
    derived_vars : dict
        Maps var_name to function name and argument list.

    Returns
    -------
    vard : TYPE
        DESCRIPTION.

    """
    dv = derived_vars[var_name]
    args = []
    for v in dv['vars']:
        if v == 'piref':
            pref = get_pref(source_dataset, ref_dataset,
                                     options)
            args.append(th.exner(pref))
        elif v == 'pref':
            pref = get_pref(source_dataset, ref_dataset,
                                     options)
            args.append(pref)
        else:
            var = get_data(source_dataset, ref_dataset, v, options,
                           allow_none=True)
            args.append(var)
    vard = dv['func'](*args)
    vard.name = var_name
    vard.attrs['units'] = dv['units']
    return vard


def get_data(source_dataset, ref_dataset, var_name: str, options: dict,
             allow_none: bool=False) :
    """
    Extract data or derived data field from source NetCDF dataset.

    If var_name is in source_dataset it is retrieved; if one of the primary
    variables with a key in var_properties the grid is corrected.
    Otherwise, it is assumed to be on a 'p' point.

    Currently written for MONC data, enforcing C-grid. Returned coords are
    'x_p', 'x_u', 'y_p', 'y_v', 'z', 'zn'. Coordinate x- and -y values are
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
    options : dict
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
        elif 'aliases' in options and var_name in options['aliases']:
            for alias in options['aliases'][var_name]:
                if alias in source_dataset:
                    print(f'Retrieving {alias:s} as {var_name:s}.')
                    vard = source_dataset[alias]
                    vard.name = var_name
                    break
            else:
                raise KeyError()
            # Change 'timeseries...' variable to 'time'
        else:
            raise KeyError()

        [itime] = get_string_index(vard.dims, ['time'])
        if itime is not None:
            vard = vard.rename({vard.dims[itime]: 'time'})

        vard = correct_grid_and_units(var_name, vard, source_dataset, options)

#        print(vard)

        if var_name == 'th' :
            thref = get_thref(ref_dataset, options)
            vard += thref

        if var_name == 'p' :
            pref = get_pref(source_dataset, ref_dataset,
                          options)
            vard += pref

    except KeyError:

        if var_name == 'thref' :
            vard = get_thref(ref_dataset, options)
        elif var_name == 'pref' :
            vard = get_pref(source_dataset, ref_dataset,
                          options)
        elif var_name == 'piref' :
            vard = th.exner(get_pref(source_dataset, ref_dataset,
                          options))
        elif var_name == 'z' :
            z = ref_dataset.dims['z']
            vard += z
        elif var_name == 'zn' :
            zn = ref_dataset.dims['zn']
            vard += zn

        elif var_name in th.derived_vars:

            vard = get_derived_vars(source_dataset, ref_dataset, options,
                                    var_name, th.derived_vars)

        else :

            if allow_none:
                vard = None
            else:
                sys.exit(f"Data {var_name:s} not in dataset.")
#    print(vard)

    return vard

def get_and_transform(source_dataset, ref_dataset, var_name, options,
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
    options : dict
        Options. Options possibly used are 'dx' and 'dy'.
    grid : str, optional
        Destination grid 'u', 'v', 'w' or 'p'. Default is 'p'.

    Returns
    -------
        var: xarray
            Output data field.

    @author: Peter Clark

    """
    var = get_data(source_dataset, ref_dataset, var_name, options)
    z = source_dataset["z"]
    zn = source_dataset["zn"]
    var = do.grid_conform(var, z, zn, grid = grid )

    # Re-chunk data if using dask
#    if not monc_utils.global_config['no_dask']:
#        var = re_chunk(var)
#    print(var)

    return var

def get_data_on_grid(source_dataset, ref_dataset, derived_dataset, var_name,
                     options, grid='p') :
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
    derived_dataset : dict
        'ds' points to xarray Dataset, 'file' to output file path.
    var_name : str
        Name of variable to retrieve.
    options : dict
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

    var_found = False
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

#    op_var = { 'name' : op_var_name }

    if options['save_all'].lower() == 'yes':

        if op_var_name in derived_dataset['ds'].variables:

            op_var = derived_dataset['ds'][op_var_name]
            print(f'Retrieved {op_var_name:s} from derived dataset.')
            var_found = True


    if not var_found:
        op_var = get_and_transform(source_dataset, ref_dataset,
                                   var_name, options, grid=grid)
        op_var.name = op_var_name

        if options['save_all'].lower() == 'yes':
            op_var = save_field(derived_dataset, op_var)
            # print(op_var)

    return op_var

def get_pref(source_dataset, ref_dataset,  options):
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
    options : dict
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

        thref = options['th_ref']

        zn = source_dataset.variables['zn'][...]
        piref0 = (p_surf/thc.p_ref_theta)**thc.kappa
        piref = piref0 - (thc.g/(thc.cp_air * thref)) * zn
        pref = thc.p_ref_theta * piref**thc.rk
#                print('pref', pref)
    else:
        pref = ref_dataset['prefn']
        [itime] = get_string_index(pref.dims, ['time'])
        if itime is not None:
            pref = pref.rename({pref.dims[itime]: 'time'})
            # tdim = pref.dims[itime]
            # pref = pref[{tdim:[0]}].squeeze()
            # pref = pref.drop_vars(tdim)

    return pref

def get_thref(ref_dataset, options):
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
        thref = options['th_ref']
    else:
        thref = ref_dataset['thref']
        [itime] = get_string_index(thref.dims, ['time'])
        if itime is not None:
            thref = thref.rename({thref.dims[itime]: 'time'})
            # tdim = thref.dims[itime]
            # thref = thref[{tdim:[0]}].squeeze()
            # thref = thref.drop_vars(tdim)

    return thref

def configure_model_resolution(dataset, options):
    """
    Find model resolution from available sources.

    This routine applies an order of precedence between potential
    pre-existing records of MONC horizontal resolutions and ensures
    that the options dictionary contains these values.

    Files written via io/dataout.py's setup_child_file will have
    the correct value listed in the file's global attributes, as this
    routine is called within that space.

    Repeated calls to this routine (for instance, to simply obtain
    dx and dy) will not modify the options contents.

    Precedence:
       1. options_database
       2. dataset attributes
       3. monc_utils options
       4. parse file path containing resolution encoding

    Parameters
    ----------
    dataset : xarray dataset
        any monc_utils-compatible dataset
    options : dict
        dataset-associated options dictionary

    Returns
    -------
    dx : float (expected)
        x-direction MONC resolution [m]
    dy : float (expected)
        y-direction MONC resolution [m]
    options : dict
        input options dictionary, possibly updated with dx and dy keys
    """
    od = options_database(dataset)
    attrs = dataset.attrs

    # 1st priority: pull from options_database, if present
    if type(od) is dict:
        dx = float(od['dxx'])
        dy = float(od['dyy'])
        options['dx'] = dx
        options['dy'] = dy
    # 2nd priority: pull from dataset attributes
    elif ('dx' in attrs and 'dy' in attrs):
        dx = attrs['dx']
        dy = attrs['dy']
        options['dx'] = dx
        options['dy'] = dy
    # 3rd priority: use values present in options
    elif ('dx' in options and 'dy' in options):
        dx = options['dx']
        dy = options['dy']
    # 4th priority: parse file path for coded info
    elif 'input_file' in options:
        dx = path_to_resolution(options['input_file'])
        dy = dx
        options['dx'] = dx
        options['dy'] = dy
    else:
        raise RuntimeError("Cannot determine grid resolution.")

    return dx, dy, options


def path_to_resolution(inpath):
    """
    Pull resolution value from an encoded path as a float.

    e.g., 'BOMEX_m0020_g0800'
    i.e., it should have '_m[0-9][0-9][0-9]' (at least 3 integers)

    Parameters
    ----------
    inpath : str
        file path

    Returns
    -------
    dx : float
        MONC horizontal resolution [m]
    """
    fullpath = inpath
    # Allow for variable length integer string.
    usc = [i for i, ltr in enumerate(fullpath) if ltr not in '0123456789']
    usc.append(len(fullpath))
    usc = np.asarray(usc)
    mnc = [m.start(0) for m in re.finditer('_m[0-9][0-9][0-9]', fullpath)][-1] + 2
    enc = usc[usc > mnc].min()
    dx = float(fullpath[mnc:enc])

    return dx
