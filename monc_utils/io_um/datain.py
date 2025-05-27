"""
Created on Mon Aug  2 11:01:11 2021.

@author: Peter Clark
"""
import numpy as np
import datetime
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.dataout import save_field
from monc_utils.data_utils.dask_utils import re_chunk

import monc_utils.data_utils.difference_ops as do
import monc_utils.thermodynamics.thermodynamics as th
import monc_utils
import re
from loguru import logger

um_datain_options = {'cartesian':False,
                     'xy_periodic':False,
                     'ref_is_init_mean':True,
                     }


stash_map = { 'u'                  : 'm01s00i002', 
              'v'                  : 'm01s00i003',
              'w'                  : 'm01s00i150',
              'th'                 : 'm01s00i004',
              'air_potential_temperature': 'm01s00i004',
              'surface_altitude'   : 'm01s00i033',
              'q_vapour'           : 'm01s00i010', 
              'specific_humidity'  : 'm01s00i010',
              'q_ice_mass'         : 'm01s00i012', 
              'q_cloud_liquid_mass': 'm01s00i254',
              'exner_rho'          : 'm01s00i255',
              'exner'              : 'm01s00i406',
              'mr_liquid_cloud'    : 'm01s00i392',
              'mr_ice cloud'       : 'm01s00i393',
              'mr_rain'            : 'm01s00i394',
              'mr_graupel'         : 'm01s00i395',
              'mr_ice crystals'    : 'm01s00i396',
              'p_rho'              : 'm01s00i407',
              'p_th'               : 'm01s00i408',
              'p'                  : 'm01s00i408',
              'p_surf'             : 'm01s00i409',
              'cloud fraction'     : 'm01s00i266',
              'rainrate'           : 'm01s04i203',
              "traj_tracer_xr"     : 'm01s00i700',
              "traj_tracer_xi"     : 'm01s00i701',
              "traj_tracer_yr"     : 'm01s00i702',
              "traj_tracer_yi"     : 'm01s00i703',
              "traj_tracer_zr"     : 'm01s00i704',
              "upward_heat_flux"   : 'm01s03i216',
              "upward_water_vapour_flux" : 'm01s03i222',
              'rho'                : 'm01s15i271',
              'u_b'                : 'm01s15i002',
              'v_b'                : 'm01s15i003',
              'air_temperature'    : 'm01s16i004',
            }

PGRID = [False,False,False]
UGRID = [True,False,False]
VGRID = [False,True,False]
WGRID = [False,False,True]
TGRID = WGRID
BGRID = [True,True,False]


def set_um_datain_options(opts):
    global um_datain_options
    um_datain_options.update(opts)
    logger.info(f'{um_datain_options=}')
    return
    
def set_um_stashmap(stash_map_update):
    global stash_map
    stash_map.update(stash_map_update)
    logger.info('Updated stash_map.')
    return
    
    
def clean_dims(field, keep_dims=None):
    
    if keep_dims is None: keep_dims = []

    for c in field.coords:
        if(c in field.dims or c == 'elapsed_time' 
           or any([d in c for d in keep_dims]) ): continue
        field = field.drop_vars(c)
        
    return field

def get_coord(field, dimname):
    
    full_dim = [d for d in field.dims if dimname in d]
    if len(full_dim) > 0 : 
        full_dim = full_dim[0]
    else:
        full_dim = None
        
    return full_dim

def coords_to_latlon(field, offsets=None):
    
    var_properties = {"u":{"grid":UGRID,
                           "units":'m.s-1'},
                      "v":{"grid":VGRID,
                           "units":'m.s-1'},
                      "w":{"grid":WGRID,
                           "units":'m.s-1'},
                      "th":{"grid":TGRID,
                            "units":'K'},
                      "p":{"grid":PGRID,
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
        
    if offsets is None:
        offsets={'x':0, 'y':0, 'z':0}
    
    var_name = field.name
    
    in_properties = var_properties.get(var_name, {"grid":TGRID,"units":''})
        
    in_grid_type = in_properties['grid']

    swap_map = {}
    for i, (coord, alt_point) in enumerate(
            zip(['longitude','latitude','model_level'],
                ['u', 'v', 'w'])):
        
        c = get_coord(field, coord)
            
        if c is not None:
            if coord in 'model_level':
                base_coord_vals = field.coords[c].values.astype("float32")
            else:
                base_coord_vals = np.arange(field.sizes[c], dtype="float32")
            if in_grid_type[i]:
                
                new_name = f'{c}_{alt_point}'
                coord_vals = base_coord_vals + offsets['xyz'[i]] + [-0.5, -0.5, 0.0][i]
                new_coord = 'xyz'[i] + f'_{alt_point}'
                
            else:
                
                new_name = f'{c}_p'
                coord_vals = base_coord_vals + offsets['xyz'[i]] + [0.0, 0.0, -0.5][i]           
                new_coord = 'xyz'[i] + '_p'
                
            field = field.assign_coords({new_coord: (c, coord_vals)})
            field = field.rename({c:new_name})
            swap_map[new_name] = new_coord

    field = field.swap_dims(swap_map)


    if 'units' not in field.attrs:        
        field.attrs['units'] = in_properties['units']
            
    return field
    
def coords_to_cartesian(field):
    
    def _adjust_cyclic_data_order(field, dimname, vname):
        logger.info(f"Rolling {vname} data")
        field = field.roll({dimname:-1}, roll_coords=True) 
        c = field.coords[old_coord].values
        c[-1] = c[-2]*2 - c[-3]
        field = field.assign_coords({old_coord:c})
        return field
        
    for new_coord, old_coord in zip(('x', 'y', 'z', 'time'), 
                                    get_um_coords(field)):
        
        if old_coord is None : continue
    
        if new_coord == 'z':
            
            vert_dim = [d for d in field.dims if 'lev_eta' in d][0]
            if 'rho' in vert_dim: new_coord_full = 'z_p'
            else: new_coord_full = 'z_w'
            
            field = field.rename({old_coord:new_coord_full})
            field = field.swap_dims({vert_dim:new_coord_full})
            
        else:
            
            new_coord_full = f'{new_coord}{get_um_grid_desc(old_coord)}'
            if new_coord in 'xy':
                
                if old_coord == 'longitude_cu':
                    field = _adjust_cyclic_data_order(field, old_coord, 'u')
                    new_coord_full = 'x_u'
                    
                if old_coord == 'latitude_cu': 
                    new_coord_full = 'y_p'
                    
                if old_coord == 'longitude_cv':
                    new_coord_full = 'x_p'
                    
                if old_coord == 'latitude_cv':
                    field = _adjust_cyclic_data_order(field, old_coord, 'v')
                    new_coord_full = 'y_v'
                    
                if field.attrs['grid_mapping'] == 'grid_crs':
                    new_coord_values = np.round(field.coords[old_coord].values  * 1000)
                else:
                    new_coord_values = field.coords[old_coord].values 
                    
                field = field.assign_coords(
                    {new_coord_full: (old_coord, new_coord_values)}) 
                   
                field = field.swap_dims({old_coord:new_coord_full})
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
            
    return field
    
def get_um_field(ds, stash:str=None, name:str=None):
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
    
    cartesian = um_datain_options.get('cartesian', True)  
    
    keep_dims = um_datain_options.get('keep_dims', [])
    
    if name is None:
        if stash is None:
            raise ValueError('No field id provided.')
        else:
            if stash in ds.data_vars:
                name = stash
            elif f'STASH_{stash}' in ds.data_vars:
                name = f'STASH_{stash}'
            else:
                raise ValueError(f'field id {stash} not in dataset.')
    field = ds[name] 
                
    if cartesian:
        field = coords_to_cartesian(field)
        field.attrs['xy_periodic'] = True
        field.attrs['cartesian'] = True
    else:
        field = coords_to_latlon(field)
        field.attrs['xy_periodic'] = False
        field.attrs['cartesian'] = False
                 
    field = clean_dims(field, keep_dims=keep_dims)
    
    return field

def get_derived_um_vars(source_dataset, 
                        var_name: str, derived_vars: dict, options: dict=None):
    """
    Get data from source_dataset and compute required variable.

    Parameters
    ----------
    source_dataset : xarray.Dataset
        Input (at least 2D) data.
    var_name : str
        Name of variable to retrieve.
    derived_vars : dict
        Maps var_name to function name and argument list.
    options : dict (optional - default=None)
        Options. Options possibly used are 'dx' and 'dy'.

    Returns
    -------
    vard : xarray.core.dataarray.DataArray
        Required data.

    """
    dv = derived_vars[var_name]
    args = []
    for v in dv['vars']:
        allow_none=False
        if v[0] == '[':
            allow_none=True
            v = v[1:-1]
        var = get_um_data(source_dataset, v, options=options,
                          allow_none=allow_none)
        if var is not None:
            args.append(var)
        else:
            logger.info(f'{v} not in dataset.')
    vard = dv['func'](*args)
    vard.name = var_name
    vard.attrs['units'] = dv['units']
    return vard
    
def get_um_data(source_dataset,
                var_name: str,
                options: dict=None,
                allow_none: bool=False) :
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
        if var_name in source_dataset:
            vard = get_um_field(source_dataset, name=var_name)
            
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
               
        if var_name == 'piref':
            vard = get_derived_um_vars(source_dataset,
                                    'exner', th.derived_vars,
                                    options=options)
            vard = get_mean(vard)
            vard.name = var_name  
            
        elif var_name[-3:] == 'ref':
            vard = get_um_data(source_dataset, var_name[:-3])
            vard = get_mean(vard)
            vard.name = var_name  
                                    
        elif var_name in th.derived_vars:

            vard = get_derived_um_vars(source_dataset, 
                                       var_name, 
                                       th.derived_vars,
                                       options=options)
            
        else:
            
            deriv = re.compile(r'dbyd[xyz]\(*')
            mo = deriv.match(var_name)
            
            if mo is not None:
        
                target_var_name = var_name[mo.end():]
                if target_var_name[-1] != ')':
                    raise KeyError(f"Data {var_name:s} not in dataset.")
                    
                target_var_name = target_var_name[:-1]
                
                target_var = get_um_data(source_dataset, 
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
                                        
                return vard
                

            else :
                if allow_none:
                    return None
                else:
                    raise KeyError(f"Data {var_name:s} not in dataset.")
                
    return vard

def get_um_and_transform(source_dataset, var_name,
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
        return clean_dims(c)
        
    
    var = get_um_data(source_dataset, var_name, options=options)
    
    z_w = _get_mapped_coord(source_dataset, 
                            'thlev_zsea_theta', 
                            'thlev_eta_theta',
                            'z_w')
    
    z_p = _get_mapped_coord(source_dataset,
                            'rholev_zsea_rho',
                            'rholev_eta_rho',
                            'z_p')
    
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
                                  var_name, options=options, grid=grid)
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
