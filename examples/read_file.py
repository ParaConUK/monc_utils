import numpy as np

import xarray as xr

import monc_utils.io.datain as din
from monc_utils.data_utils.string_utils import get_string_index

indir = '/storage/silver/wxproc/xm904103/hy913279/'
infile = 'BOMEX_m0025_g0600_all_88200.0.nc'

ds = xr.open_dataset(indir+infile)

print(ds)

th = din.get_data(ds, None, 'th')
z = din.get_data(ds, None, 'z')
z_p = din.get_data(ds, None, 'z_p')

print(th)


thref = din.get_thref(None)
print(thref)

print(th.time)