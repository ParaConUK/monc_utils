"""
difference_ops.py.

Created on Wed Apr 17 21:03:43 2019

Difference operators for C-grid data.

Note: written for MONC grid

v[i  ,j  ,k] -- +            -- v[i+1,j  ,k] -- +   
|               |               |               |
|               |               |               |
p[i  ,j  ,k] -- u[i  ,j  ,k] -- p[i+1,j  ,k] -- u[i+1,j,k]   
|               |               |               |
|               |               |               |

v[i  ,j-1,k] -- +            -- v[i+1,j-1,k] -- +   
   
The 0th point is a p point. We have decided this is at dx/2, dy/2 

roll(f, +1) shifts data right, so is equivalent to f[i-1] (or j-1).

For optional UM grid:
+           --  v[i ,j+1,k]  -- +            -- v[i+1 ,j  ,k] --+   
|               |               |               |
|               |               |               |
u[i  ,j  ,k]-- p[i  ,j  ,k]  -- u[i+1,j  ,k] -- p[i+1,j  ,k] -- u[i+2,j,k]   
|               |               |               |
|               |               |               |
+           -- v[i  ,j  ,k]  -- +            -- v[i+1 ,j  ,k] --+   
   
The 0th point is a p point. We have decided this is at dx/2, dy/2 

roll(f, +1) shifts data right, so is equivalent to f[i-1] (or j-1).


@author: Peter Clark
"""
import numpy as np
import monc_utils
from .dask_utils import re_chunk
from .string_utils import get_string_index
import xarray
import typing
import warnings

from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning,
                                  module='xarray.core.missing')

difference_ops_options = {'cartesian':False,
                          'xy_periodic':False,
                          'UM_grid':False,
                          # 'UM_grid':True,
                         }

grid_def = { 'p':('x_p', 'y_p', 'z_p'),
             'u':('x_u', 'y_p', 'z_p'),
             'v':('x_p', 'y_v', 'z_p'),
             'w':('x_p', 'y_p', 'z_w')}

def set_difference_ops_options(opts):
    global difference_ops_options
    difference_ops_options.update(opts)
    logger.info(f'{difference_ops_options=}')
    return


def exec_fn(fn: typing.Callable, 
            field: xarray.DataArray, axis: int) -> xarray.DataArray:
    """
    Execute function using map_overlap with overlap on selected axis.

    Parameters
    ----------
    fn : function
        DESCRIPTION.
    field : xarray.DataArray
        DESCRIPTION.
    axis : int
        DESCRIPTION.

    Returns
    -------
    new xarray.DataArray

    """
#    print(type(field.data))
    if monc_utils.global_config['no_dask'] or type(field.data) is np.ndarray:
        field = fn(field)
    else:
        if monc_utils.global_config['use_map_overlap']:
#            print('Using map_overlap.')
            d = field.data.map_overlap(fn, depth={axis:1},
                                      boundary={axis:'periodic'})
            field.data = d
        else:
            sh = np.shape(field)
            ch = {field.dims[axis]:sh[axis]}
            field = re_chunk(field, chunks=ch)
            field = fn(field)
    return field


def last_dim(z) :
    """
    Remove all but last dimension of z. (Deprecated)

    Parameters
    ----------
        z : n-dimensional array.

    Returns
    -------
        last dimension of z.
        
    @author: Peter Clark
    """
    zd = z[...]
    while len(np.shape(zd))>1 :
        zd = zd[0,...]
    return zd

def interpolate_z(field, znew) :
    """
    Interpolate field znew.

    Parameters
    ----------
        field : xarray nD field
        znew  : xarray coordinate new z.

    Returns
    -------
        field on znew levels
    @author: Peter Clark
    """
    [zaxis] = get_string_index(field.dims,['z'])
    zdim = field.dims[zaxis]

    newfield = field.interp({zdim:znew}, kwargs={"fill_value": "extrapolate"})
    newfield = newfield.drop_vars(zdim)

    return newfield

def interp_aux_coords(field, dim, newfield, target_dim):
    """
    Intorpolat non-dimensional coords to new dimension.

    Parameters
    ----------
    field : xarray.DataArray
        nD field.
    dim : char
        dimension in field. e.g. 'x_p'
    newfield : xarray.DataArray
        field with new aux_coords associated with dim.
    target_dim : char
        new dimension in field. e.g. 'x_v'

    Returns
    -------
    newfield : xarray.DataArray
        copy of field with target_dim coords.
    """
    
    for coord in field.coords:
        if coord == dim: continue
        if dim not in field.coords[coord].dims : continue
        newcoord = field.coords[coord].interp(
                      {dim:newfield.coords[target_dim]},
                      assume_sorted=True,
                      kwargs={'fill_value':'extrapolate'} )
        newcoord.name = coord[:-2] + target_dim[1:]
        newcoord = newcoord.drop_vars([dim, coord])
        newfield = newfield.drop_vars(coord)
        newfield = newfield.assign_coords({newcoord.name:newcoord})
        
    return newfield

def grid_conform_xy(field, target_dim):
    """
    Force field to target grid by averaging if necessary.
    This works on x or y grid.
    This replaces grid_conform_x and grid_conform_y

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with x dimension 'x_u' or 'x_p'. Any other
        x dimensionm treated as 'x_p'.
        OR 
        Any multi-dimensional xarray with y dimension 'y_v' or 'y_p'. Any other
        y dimensionm treated as 'y_p'.
    target_xdim : str
       Dimension name 'x_u' or 'x_p' OR 'y_v' or 'y_p'

    Returns
    -------
    xarray
        field on target x or y grid.

    """
    um_grid = difference_ops_options['UM_grid']

    dimname = target_dim[0]
    
    [axis_number] = get_string_index(field.dims,[dimname])

    if axis_number is None:
        logger.info(f'{field.name} no {dimname} axis in field dims.')
        return field

    dim = field.dims[axis_number]

    if dim == target_dim:
        logger.info(f'{field.name} {dimname} is already {target_dim}')
        return field

    match (target_dim, um_grid):
        case ('x_p', True) | ('x_u', False) | ('y_p', True) | ('y_v', False):
            # Data on x_u will have (f[i+1] + f[i])/2 on x_p[i]
            rollval = -1
        case ('x_u', True) | ('x_p', False) | ('y_v', True) | ('y_p', False) :
            # Data on x_u will have (f[i] + f[i-1])/2 on x_p[i]
            rollval = +1
        case _:
            logger.warning(f"Cannot transform {dim} to {target_dim}")
            return field
        
    dim_coord_values = field.coords[dim].data
    dim_coord_spacing = dim_coord_values[1] - dim_coord_values[0]
    dim_coord_values_new = dim_coord_values - rollval * dim_coord_spacing / 2.0

    logger.info(f'{field.name} {dim} to {target_dim}')
    
    if difference_ops_options['xy_periodic']:
        mn  = lambda arr:(0.5 
                          * (arr + np.roll(arr, rollval, axis=axis_number)))
        newfield = field.rename({dim:target_dim})
        newfield = exec_fn(mn, newfield, axis_number)
        newfield.coords[target_dim] = dim_coord_values_new
        
    else:
        
        newfield = field.interp({dim:dim_coord_values_new},
                                assume_sorted=True,
                                kwargs={'fill_value':'extrapolate'} )
        newfield = newfield.rename({dim:target_dim})
    
    newfield = interp_aux_coords(field, dim, newfield, target_dim)

    return newfield

def grid_conform_x(field, target_xdim):
    """
    Force field to target x grid by averaging if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with x dimension 'x_u' or 'x_p'. Any other
        x dimensionm treated as 'x_p'.
    target_xdim : str
       Dimension name 'x_u' or 'x_p'

    Returns
    -------
    xarray
        field on target x grid.

    """
    logger.warning('grid_conform_x is deprecated - use grid_conform_xy')
    newfield = grid_conform_xy(field, target_xdim)
    return newfield

def grid_conform_y(field, target_ydim):
    """
    Force field to target y grid by averaging if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with y dimension 'y_v' or 'y_p'. Any other
        y dimensionm treated as 'y_p'.
    target_xdim : str
       Dimension name 'y_v' or 'y_p'

    Returns
    -------
    xarray
        field on target y grid.

    """
    logger.warning('grid_conform_x is deprecated - use grid_conform_xy')
    newfield = grid_conform_xy(field, target_ydim)
    return newfield

def grid_conform_z(field, z_w, z_p, target_zdim):
    """
    Force field to target x grid by interpolation if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with z dimension 'z_w' or 'z_p'.
    z_w : xarray coord.
    z_p : xarray coord.
    target_xdim : str
       Dimension name 'z_w' or 'z_p'

    Returns
    -------
    xarray
        field on target x grid.

    """
    [zaxis] = get_string_index(field.dims,['z'])
    if zaxis is None:
        return field
    zdim = field.dims[zaxis]
    if zdim == target_zdim:
        logger.info(f'{field.name} zdim is already {target_zdim}')
        return field
    elif target_zdim == 'z_w':
        logger.info(f'{field.name} {zdim} to {target_zdim}')
        newfield = interpolate_z(field, z_w)
    elif target_zdim == 'z_p':
        logger.info(f'{field.name} {zdim} to {target_zdim}')
        newfield = interpolate_z(field, z_p)
    else:
        logger.warning(f"{field.name}: cannot transform {zdim} to {target_zdim}")
        return field

    newfield = interp_aux_coords(field, zdim, newfield, target_zdim)
        
    return newfield

def grid_conform(field, z_w, z_p, grid: str = 'p' ):
    """
    Force field to target grid by interpolation if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with z dimension 'z_w' or 'z_p'.
    z_w : xarray coord.
    z_p : xarray coord.
    grid : str | tuple(str)
       grid identifier 'p'| 'u'| 'v'| 'w' or tuple (xdim, ydim, zdim).

    Returns
    -------
    xarray
        field on target grid.

    """
        
    if type(grid) == str:
        if grid in ['p', 'u', 'v', 'w']:
            op_grid = grid_def[grid]
        else:
            raise ValueError(f"grid={grid} is illegal value.")
    else:
        op_grid = grid

    newfield = grid_conform_xy(field, op_grid[0])
    newfield = grid_conform_xy(newfield, op_grid[1])
    newfield = grid_conform_z(newfield, z_w, z_p, op_grid[2])
    return newfield

def d_by_dxy_field_native(field, dim_dir):
    """
    Differentiate field in x direction on native grid.

    Parameters
    ----------
        field : xarray nD field

    Returns
    -------
        field on native grid

    """
    
    um_grid = difference_ops_options['UM_grid']

    dual_grid_names = {'x_p':'x_u', 'x_u':'x_p', 'y_p':'y_v', 'y_v':'y_p'}

    [axis_number] = get_string_index(field.dims,[dim_dir])
    dim = field.dims[axis_number]

    if dim not in dual_grid_names:
        logger.warning(f"d_by_d{dim_dir}_field on unknown grid {dim},"
                       " assuming {dim_dir}_p.")
        field = field.rename({dim:f'{dim_dir}_p'})
        
        
    dim_coord_values = field.coords[dim].data
    
    match (dim, um_grid):
        case ('x_p', True) | ('x_u', False) | ('y_p', True) | ('y_v', False):
            # Data on x_u will have (f[i+1] - f[i])/2 
            rollval = +1
        case ('x_u', True) | ('x_p', False) | ('y_v', True) | ('y_p', False) :
            # Data on x_u will have (f[i] - f[i-1])/2 
            rollval = -1
        case _:
            logger.warning(f"Cannot differentiate wrt {dim}")
            return field
    
    dim_new = dual_grid_names[dim]
    
    if difference_ops_options['xy_periodic']:
        dim_coord_spacing = dim_coord_values[1] - dim_coord_values[0]
        drv = lambda arr:(rollval 
                          * (arr - np.roll(arr,  rollval, axis=axis_number))
                          / dim_coord_spacing)
        dim_coord_values_new = (dim_coord_values 
                                - rollval * dim_coord_spacing / 2.0)
        newfield = exec_fn(drv, field.copy(), axis_number)
    
    else:
        
        dim_coord_values_new = 0.5 * (dim_coord_values[1:] 
                                    + dim_coord_values[0:-1])
        dim_coord_spacing = (dim_coord_values[1:] 
                              - dim_coord_values[0:-1])
        newfield = field.diff(dim) 
        
        dim_coord_spacing = xarray.DataArray(dim_coord_spacing, 
                                coords = {dim:newfield.coords[dim].values})
        
        newfield = newfield / dim_coord_spacing
        
    newfield = newfield.rename({dim:dim_new})
    newfield.coords[dim_new] = dim_coord_values_new
    newfield.name = f"dbyd{dim_dir}({field.name:s})"
    
    if 'units' in field.attrs:
        newfield.attrs['units'] = field.attrs['units'] + '.m-1'
    
    newfield = interp_aux_coords(field, dim, newfield, dim_new)

    logger.info(f"d_by_d{dim_dir}_{field.name}_on_{dim_new} ")

    return newfield

def d_by_dx_field_native(field):
    """
    Differentiate field in x direction on native grid.

    Parameters
    ----------
        field : xarray nD field

    Returns
    -------
        field on native grid

    """
    logger.warning('d_by_dx_field_native is deprecated '
                   '- use d_by_dxy_field_native')
    newfield = d_by_dxy_field_native(field, 'x')
    return newfield


def d_by_dx_field(field, z_w, z_p, grid: str = 'p' ) :
    """
    Differentiate field in x direction.

    Parameters
    ----------
        field : xarray nD field
        z_w: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        z_p: xarray coordinate
            zcoord on p levels - needed if changing vertical grid.
        grid : str | tuple of 2 strings
            destination grid (Default = 'p')

    Returns
    -------
        field on required grid

    @author: Peter Clark
    """
    logger.warning('d_by_dx_field is deprecated - use d_by_dxy_field')
    newfield = d_by_dxy_field_native(field, 'x')
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"dbydx({field.name:s})_on_{grid:s}"

    return newfield

def d_by_dy_field_native(field):
    """
    Differentiate field in y direction on native grid.

    Parameters
    ----------
        field : xarray nD field

    Returns
    -------
        field on native grid

    """
    logger.warning('d_by_dx_field_native is deprecated '
                   '- use d_by_dxy_field_native')
    newfield = d_by_dxy_field_native(field, 'y')

    return newfield   

def d_by_dy_field(field, z_w, z_p, grid: str = 'p' ) :
    """
    Differentiate field in y direction.

    Parameters
    ----------
        field : xarray nD field
        z_w: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        z_p: xarray coordinate
            zcoord on p levels - needed if changing vertical grid.
        grid : str | tuple of 2 strings
            destination grid (Default = 'p')

    Returns
    -------
        field on required grid

    @author: Peter Clark
    """
    logger.warning('d_by_dy_field is deprecated - use d_by_dxy_field')
    newfield = d_by_dxy_field_native(field, 'y')
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"dbydy({field.name:s})_on_{grid:s}"

    return newfield

def d_by_dxy_field(field, z_w, z_p, dim_dir, grid: str = 'p' ) :
    """
    Differentiate field in x direction.

    Parameters
    ----------
        field : xarray nD field
        z_w: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        z_p: xarray coordinate
            zcoord on p levels - needed if changing vertical grid.
        dim_dir: char
            Direction to differentiate: 'x' or 'y'.
        grid : str | tuple of 2 strings
            destination grid (Default = 'p')

    Returns
    -------
        field on required grid

    @author: Peter Clark
    """
    newfield = d_by_dxy_field_native(field, dim_dir)
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"dbyd{dim_dir}({field.name:s})_on_{grid:s}"

    return newfield

def d_by_dz_field_native(field):
    """
    Differentiate field in z direction on native grid.

    Parameters
    ----------
        field : xarray nD field

    Returns
    -------
        field on native grid

    """
    [zaxis] = get_string_index(field.dims,['z'])
    zdim = field.dims[zaxis]
    zcoord = field.coords[zdim].data
    if zdim == 'z_p' or zdim == 'zn' :
        logger.info(f"d_by_dz_{field.name}_on_z_p ")
        # Differences will be at midpoints between z_p points.
        # These are only z_w points on a uniform grid.
        # Furthermore, we need additional point at the top.
        zdim_new = 'z_i'
        pad = (0,1)
        nroll = -1
        (exn, dexn) = (-1, -1)
        
    elif zdim == 'z' :
        logger.info(f"d_by_dz_{field.name}_on_z ")
        zdim_new = 'zn'
        pad = (1,0)
        nroll = 1
        (exn, dexn) = (0, 1)
        
    else:
        logger.info(f"d_by_dz_{field.name}_on_z_w ")
        # Differences will be at midpoints between z_w points.
        # These are z_p points even on a uniform grid.
        # We need additional point at the bottom.
        zdim_new = 'z_p'
        pad = (1,0)
        nroll = 1
        (exn, dexn) = (0, 1)

    newfield = field.diff(zdim)/field.coords[zdim].diff(zdim)
    newfield = newfield.pad(pad_width={zdim:pad}, mode = 'edge')
    newfield = newfield.rename({zdim:zdim_new})
    
    z_i = 0.5 * (zcoord + np.roll(zcoord, nroll))
    z_i[exn] = 2 * z_i[exn + dexn] - z_i[exn + 2 * dexn]
    
    newfield.coords[zdim_new] = z_i
    newfield.loc[{zdim_new:z_i[exn]}] = ( 2 * newfield.isel({zdim_new:exn + dexn}) 
                                     - newfield.isel({zdim_new:exn + 2 * dexn}) )
                                   
    newfield.name = f"dbydz({field.name:s})"
    if 'units' in field.attrs:
        newfield.attrs['units'] = field.attrs['units'] + '.m-1'

    newfield = interp_aux_coords(field, zdim, newfield, zdim_new)

    return newfield

def d_by_dz_field(field, z_w, z_p, grid: str = 'p'):
    """
    Differentiate field in z direction.

    Parameters
    ----------
        field : xarray nD field
        z_w: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        z_p: xarray coordinate
            zcoord on p levels - needed if changing vertical grid.
        grid : str | tuple of 2 strings
            destination grid (Default = 'p')

    Returns
    -------
        field on required grid

    @author: Peter Clark
    """

    newfield = d_by_dz_field_native(field)
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"d{field.name:s}_by_dz_on_{grid:s}"

    return newfield


def padleft(f, zt, axis=0) :
    """
    Add dummy field at bottom of nD array.

    Parameters
    ----------
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns
    -------
        extended field, extended coord
    @author: Peter Clark
    """
    s = list(np.shape(f))
    s[axis] += 1
    newfield = np.zeros(s)
    newfield[...,1:]=f
    newz = np.zeros(np.size(zt)+1)
    newz[1:] = zt
    newz[0] = 2*zt[0]-zt[1]
    return newfield, newz

def padright(f, zt, axis=0) :
    """
    Add dummy field at top of nD array.

    Parameters
    ----------
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns
    -------
        extended field, extended coord
    @author: Peter Clark
    """
    s = list(np.shape(f))
    s[axis] += 1
    newfield = np.zeros(s)
    newfield[...,:-1] = f
    newz = np.zeros(np.size(zt)+1)
    newz[:-1] = zt
    newz[-1] = 2*zt[-1]-zt[-2]
    return newfield, newz
