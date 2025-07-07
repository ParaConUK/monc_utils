import numpy as np
import xarray as xr
from monc_utils.data_utils.difference_ops import (difference_ops_options)


def create_uniform_grid(dL, L):
    """
    Create a grid with uniform resolution in x,y and z with domain spanning
    [0,0,0] to `L` with grid resolution `dL`


    """
    Lx, Ly, Lz = L
    dx, dy, dz = dL


    x_p = np.linspace(dx / 2.0, Lx - dx / 2.0, round(Lx/dx))
    y_p = np.linspace(dy / 2.0, Ly - dy / 2.0, round(Ly/dy))
    if difference_ops_options['xy_periodic']:
        # create wrapped positions starting at 0.
        if difference_ops_options['UM_grid']:
            x_u = np.linspace(0, Lx - dx , round(Lx/dx))
            y_v = np.linspace(0, Ly - dy, round(Ly/dy))
        else:
            x_u = np.linspace(dx, Lx , round(Lx/dx))
            y_v = np.linspace(dy, Ly, round(Ly/dy))
    else:
        # if difference_ops_options['UM_grid']:
        x_u = np.linspace(0, Lx, round(Lx/dx))
        y_v = np.linspace(0, Ly, round(Ly/dy))
        # else:
        #     x_u = np.linspace(dx, Lx , round(Lx/dx))
        #     y_v = np.linspace(dy, Ly, round(Ly/dy))
        
    z_p = np.linspace(-dz / 2.0, Lz - dz / 2.0, round(Lz/dz)+1)
    z_w = np.linspace(0, Lz, round(Lz/dz)+1)

    ds = xr.Dataset(coords=dict(x_p=x_p, x_u=x_u, y_p=y_p, y_v=y_v, z_p=z_p,
                                z_w=z_w))
    
    return ds


def create_initial_dataset(dL, L, xy_periodic=True, delta_lat=10.0, 
                           delta_lon=5.0, delta_height = 100.0):
    """
    Create an initial dataset with a uniform grid and the position scalars
    initiated to the locations in the grid.
    """
    ds = create_uniform_grid(dL=dL, L=L)

    ds.attrs["xy_periodic"] = xy_periodic
    
    X , Y, Z = xr.broadcast(ds['x_p'], ds['y_p'], ds['z_p'])

    ds["fx"] = np.sin(2.0 * np.pi * X / L[0])
    ds["dfxdx"] = np.cos(2.0 * np.pi * X / L[0]) * (2.0 * np.pi / L[0])
    ds["fy"] = np.sin(2.0 * np.pi * Y / L[1])
    ds["dfydy"] = np.cos(2.0 * np.pi * Y / L[1]) * (2.0 * np.pi / L[1])
    ds["fz"] = (Z / L[2]) ** 2
    ds["dfzdz"] = 2.0 * Z / L[2] ** 2
    ds.assign_coords(time = np.datetime64("2020-01-01T00:00"))
    
    ds = ds.assign_coords(
                     {'grid_longitude_p': ('x_p', ds.x_p.values * delta_lat),
                      'grid_latitude_p': ('y_p', ds.y_p.values * delta_lon),
                      'height_p': ('z_p', ds.z_p.values * delta_height),})
    
    print(ds)

    return ds


