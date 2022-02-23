"""
difference_ops.py.

Created on Wed Apr 17 21:03:43 2019

Difference operators for C-grid data.

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
            print('Using map_overlap.')
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

def interpolate(field, znew) :
    """
    Interpolate field from z to zn.

    Parameters
    ----------
        field : xarray nD field
        znew  : xarray coordinate new z.

    Returns
    -------
        field on zn levels
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
        xmn = lambda arr:(0.5 * (arr + np.roll(arr, -1, axis=xaxis)))
        x_new = x - dx / 2.0
    elif target_xdim == 'x_u':
        xmn = lambda arr:(0.5 * (arr + np.roll(arr, +1, axis=xaxis)))
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
        ymn = lambda arr:(0.5 * (arr + np.roll(arr, -1, axis=yaxis)))
        y_new = y - dy / 2.0
    elif target_ydim == 'y_v':
        ymn = lambda arr:(0.5 * (arr + np.roll(arr, +1, axis=yaxis)))
        y_new = y + dy / 2.0
    else:
        print(f"Cannot transform {ydim} to {target_ydim}")
        return field

    print(f'{field.name} {ydim} to {target_ydim}')
    newfield = field.rename({ydim:target_ydim})
    newfield = exec_fn(ymn, newfield, yaxis)
    newfield.coords[target_ydim] = y_new
    return newfield

def grid_conform_z(field, z, zn, target_zdim):
    """
    Force field to target x grid by interpolation if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with z dimension 'z' or 'zn'.
    z : xarray coord.
    zn : xarray coord.
    target_xdim : str
       Dimension name 'z' or 'z'

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
        return interpolate(field, z)
    elif target_zdim == 'z_p':
        print(f'{field.name} {zdim} to {target_zdim}')
        return interpolate(field, zn)
    else:
        print(f"{field.name}: cannot transform {zdim} to {target_zdim}")
        return field

def grid_conform(field, z, zn, grid: str = 'p' ):
    """
    Force field to target grid by interpolation if necessary.

    Parameters
    ----------
    field : xarray
        Any multi-dimensional xarray with z dimension 'z' or 'zn'.
    z : xarray coord.
    zn : xarray coord.
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
    newfield = grid_conform_z(newfield, z, zn, op_grid[2])
    return newfield

def d_by_dx_field(field, z, zn, grid: str = 'p' ) :
    """
    Differentiate field in x direction.

    Parameters
    ----------
        field : xarray nD field
        z: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        zn: xarray coordinate
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
        xdim_new = 'x_p'
        xdrv = lambda arr:((arr - np.roll(arr,  1, axis=xaxis)) / dx)
        x_new = x - dx / 2.0
    else:
        if xdim != 'x_p':
            print(f"d_by_dx_field on unknown grid {xdim}, assuming x_p.")
        print("d_by_dx_field_on_x_p ",grid)
        xdim_new = 'x_u'
        xdrv = lambda arr:((np.roll(arr, -1, axis=xaxis) - arr) / dx)
        x_new = x + dx / 2.0

    newfield = field.rename({xdim:xdim_new})
    newfield = exec_fn(xdrv, newfield, xaxis)
    newfield.coords[xdim_new] = x_new
    newfield = grid_conform(newfield, z, zn, grid=grid)
    newfield.name = f"d{field.name:s}_by_dx_on_{grid:s}"

    return newfield

def d_by_dy_field(field, z, zn, grid: str = 'p' ) :
    """
    Differentiate field in y direction.

    Parameters
    ----------
        field : xarray nD field
        z: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        zn: xarray coordinate
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
        ydim_new = 'y_p'
        ydrv = lambda arr:((arr - np.roll(arr,  1, axis=yaxis)) / dy)
        y_new = y - dy / 2.0
    else:
        if ydim != 'y_p':
            print(f"d_by_dy_field on unknown grid {ydim}, assuming y_p.")
        print("d_by_dy_field_on_y_p ",grid)
        ydim_new = 'y_v'
        ydrv = lambda arr:((np.roll(arr, -1, axis=yaxis) - arr) / dy)
        y_new = y + dy / 2.0

    newfield = field.rename({ydim:ydim_new})
    newfield = exec_fn(ydrv, newfield, yaxis)
    newfield.coords[ydim_new] = y_new
    newfield = grid_conform(newfield, z, zn, grid=grid)
    newfield.name = f"d{field.name:s}_by_dy_on_{grid:s}"

    return newfield

def d_by_dz_field(field, z, zn, grid: str = 'p'):
    """
    Differentiate field in z direction.

    Parameters
    ----------
        field : xarray nD field
        z: xarray coordinate
            zcoord on w levels - needed if changing vertical grid.
        zn: xarray coordinate
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
        zdim_new = 'zi'
        pad = (0,1)
        nroll = -1
        (exn, dexn) = (-1, -1)
    else:
        print("d_by_dz_field_on_z_w ")
        zdim_new = 'z'
        pad = (1,0)
        nroll = 1
        (exn, dexn) = (0, 1)

    newfield = field.diff(zdim)/field.coords[zdim].diff(zdim)
    newfield = newfield.pad(pad_width={zdim:pad}, mode = 'edge')
    newfield = newfield.rename({zdim:zdim_new})
    zi = 0.5 * (zcoord + np.roll(zcoord, nroll))
    zi[exn] = 2 * zi[exn + dexn] - zi[exn + 2 * dexn]
    newfield.coords[zdim_new] = zi
    newfield = grid_conform(newfield, z, zn, grid=grid)
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
