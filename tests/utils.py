import numpy as np
import xarray as xr


def create_uniform_grid(dL, L):
    """
    Create a grid with uniform resolution in x,y and z with domain spanning
    [0,0,0] to `L` with grid resolution `dL`


    """
    Lx, Ly, Lz = L
    dx, dy, dz = dL

    # create wrapped positions starting at 0.
    x_p = np.linspace(dx / 2.0, Lx - dx / 2.0, round(Lx/dx))
    x_u = np.linspace(dx, Lx , round(Lx/dx))
    y_p = np.linspace(dy / 2.0, Ly - dy / 2.0, round(Ly/dy))
    y_v = np.linspace(dy, Ly, round(Ly/dy))
    z_p = np.linspace(-dz / 2.0, Lz - dz / 2.0, round(Lz/dz)+1)
    z_w = np.linspace(0, Lz, round(Lz/dz)+1)

    ds = xr.Dataset(coords=dict(x_p=x_p, x_u=x_u, y_p=y_p, y_v=y_v, z_p=z_p,
                                z_w=z_w))

    return ds


def create_initial_dataset(dL, L, xy_periodic=True):
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
    
    print(ds)

    return ds


