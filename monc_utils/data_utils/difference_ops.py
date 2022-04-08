"""
difference_ops.py.

Created on Wed Apr 17 21:03:43 2019

Difference operators for C-grid data.

Note: written for MONC grid:
   v[i  ,j  ,k] -- +            -- v[i+1 ,j  ,k] --+   
   |               |               |               |
   |               |               |               |
-- p[i  ,j  ,k] -- u[i  ,j  ,k] -- p[i+1,j  ,k] -- u[i+1,j,k]   
   |               |               |               |
   |               |               |               |
   
The 0th point is a p point. We have decided this is at dx/2, dy/2 

roll(f, +1) shifts data right, so is equivalent to f[i-1] (or j-1).
 

@author: Peter Clark
"""
import numpy as np
import monc_utils
from .dask_utils import re_chunk
from .string_utils import get_string_index
import xarray

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                                   module='xarray.core.missing')


grid_def = { 'p':('x_p', 'y_p', 'z_p'),
             'u':('x_u', 'y_p', 'z_p'),
             'v':('x_p', 'y_v', 'z_p'),
             'w':('x_p', 'y_p', 'z_w')}


def exec_fn(fn, field: xarray.DataArray, axis: int) -> xarray.DataArray:
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
    if monc_utils.global_config['no_dask']:
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
    Remove all but last dimension of z.

    Parameters
    ----------
        z : n-dimensional array.

    Returns
    -------
        z[0,0, etc. ,:]
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
    [xaxis] = get_string_index(field.dims,['x'])
    if xaxis is None:
        return field
    xdim = field.dims[xaxis]
    if xdim == target_xdim:
        print(f'{field.name} xdim is already {target_xdim}')
        return field
    x = field.coords[xdim].data
    dx = x[1] - x[0]
    if target_xdim == 'x_p':
        # Data on x_u will have (f[i] + f[i-1])/2 on x_p[i]
        xmn = lambda arr:(0.5 * (arr + np.roll(arr, +1, axis=xaxis)))
        x_new = x - dx / 2.0
    elif target_xdim == 'x_u':
        # Data on x_p will have (f[i] + f[i+1])/2 on x_u[i]
        xmn = lambda arr:(0.5 * (arr + np.roll(arr, -1, axis=xaxis)))
        x_new = x + dx / 2.0
    else:
        print(f"Cannot transform {xdim} to {target_xdim}")
        return field

    print(f'{field.name} {xdim} to {target_xdim}')
    newfield = field.rename({xdim:target_xdim})
    newfield = exec_fn(xmn, newfield, xaxis)
    newfield.coords[target_xdim] = x_new
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
    [yaxis] = get_string_index(field.dims,['y'])
    if yaxis is None:
        return field
    ydim = field.dims[yaxis]
    if ydim == target_ydim:
        print(f'{field.name} ydim is already {target_ydim}')
        return field
    y = field.coords[ydim].data
    dy = y[1] - y[0]
    if target_ydim == 'y_p':
        # Data on y_v will have (f[j] + f[j-1])/2 on y_p[j]
        ymn = lambda arr:(0.5 * (arr + np.roll(arr, +1, axis=yaxis)))
        y_new = y - dy / 2.0
    elif target_ydim == 'y_v':
        # Data on y_p will have (f[j] + f[j+1])/2 on y_v[j]
        ymn = lambda arr:(0.5 * (arr + np.roll(arr, -1, axis=yaxis)))
        y_new = y + dy / 2.0
    else:
        print(f"Cannot transform {ydim} to {target_ydim}")
        return field

    print(f'{field.name} {ydim} to {target_ydim}')
    newfield = field.rename({ydim:target_ydim})
    newfield = exec_fn(ymn, newfield, yaxis)
    newfield.coords[target_ydim] = y_new
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
        print(f'{field.name} zdim is already {target_zdim}')
        return field
    elif target_zdim == 'z_w':
        print(f'{field.name} {zdim} to {target_zdim}')
        return interpolate_z(field, z_w)
    elif target_zdim == 'z_p':
        print(f'{field.name} {zdim} to {target_zdim}')
        return interpolate_z(field, z_p)
    else:
        print(f"{field.name}: cannot transform {zdim} to {target_zdim}")
        return field

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

    newfield = grid_conform_x(field, op_grid[0])
    newfield = grid_conform_y(newfield, op_grid[1])
    newfield = grid_conform_z(newfield, z_w, z_p, op_grid[2])
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
    [xaxis] = get_string_index(field.dims,['x'])
    xdim = field.dims[xaxis]
    x = field.coords[xdim].data
    dx = x[1] - x[0]
    if xdim == 'x_u':
        print("d_by_dx_field_on_x_u ", grid)
        # Data on x_u will have (f[i] - f[i-1])/dx on x_p[i]
        xdim_new = 'x_p'
        xdrv = lambda arr:((arr - np.roll(arr,  1, axis=xaxis)) / dx)
        x_new = x - dx / 2.0
    else:
        if xdim != 'x_p':
            print(f"d_by_dx_field on unknown grid {xdim}, assuming x_p.")
        print("d_by_dx_field_on_x_p ",grid)
        # Data on x_p will have (f[i+1] - f[i])/dx on x_u[i]
        xdim_new = 'x_u'
        xdrv = lambda arr:((np.roll(arr, -1, axis=xaxis) - arr) / dx)
        x_new = x + dx / 2.0

    newfield = field.rename({xdim:xdim_new})
    newfield = exec_fn(xdrv, newfield, xaxis)
    newfield.coords[xdim_new] = x_new
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"d{field.name:s}_by_dx_on_{grid:s}"

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
    [yaxis] = get_string_index(field.dims,['y'])
    ydim = field.dims[yaxis]
    y = field.coords[ydim].data
    dy = y[1] - y[0]
    if ydim == 'y_v':
        print("d_by_dy_field_on_y_v ", grid)
        # Data on y_v will have (f[j] - f[j-1])/dy on y_p[j]
        ydim_new = 'y_p'
        ydrv = lambda arr:((arr - np.roll(arr,  1, axis=yaxis)) / dy)
        y_new = y - dy / 2.0
    else:
        if ydim != 'y_p':
            print(f"d_by_dy_field on unknown grid {ydim}, assuming y_p.")
        print("d_by_dy_field_on_y_p ",grid)
        # Data on y_p will have (f[j+1] - f[j])/dy on y_v[j]
        ydim_new = 'y_v'
        ydrv = lambda arr:((np.roll(arr, -1, axis=yaxis) - arr) / dy)
        y_new = y + dy / 2.0

    newfield = field.rename({ydim:ydim_new})
    newfield = exec_fn(ydrv, newfield, yaxis)
    newfield.coords[ydim_new] = y_new
    newfield = grid_conform(newfield, z_w, z_p, grid=grid)
    newfield.name = f"d{field.name:s}_by_dy_on_{grid:s}"

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
    [zaxis] = get_string_index(field.dims,['z'])
    zdim = field.dims[zaxis]
    zcoord = field.coords[zdim].data
    if zdim == 'z_p':
        print("d_by_dz_field_on_z_p ")
        # Differences will be at midpoints between z_p points.
        # These are only z_w points on a uniform grid.
        # Furthermore, we need additional point at the top.
        zdim_new = 'zi'
        pad = (0,1)
        nroll = -1
        (exn, dexn) = (-1, -1)
    else:
        print("d_by_dz_field_on_z_w ")
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
    
    zi = 0.5 * (zcoord + np.roll(zcoord, nroll))
    zi[exn] = 2 * zi[exn + dexn] - zi[exn + 2 * dexn]
    newfield.coords[zdim_new] = zi
    
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
