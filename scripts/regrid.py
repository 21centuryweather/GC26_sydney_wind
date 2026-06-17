import numpy as np
import pandas as pd
import xarray as xr

import xesmf as xe

diri = "/g/data/gb02/mjl561/um2nc/SY_djf/SY_11p1/CTRL/"
diro = "/scratch/gb02/mr4682/GC26_sydney_wind/um2nc/SY_djf/SY_11p1/CTRL/"

var_name = "wnd_ucmp"

file_old = f"{diri}{var_name}-SY_SY_11p1_CTRL-v1-201611300100-201703010000.nc"
file_ref = f"{diri}sfc_temp-SY_SY_11p1_CTRL-v1-201611300100-201703010000.nc"
file_new = f"{diro}{var_name}-SY_SY_11p1_CTRL-v1-201611300100-201703010000.nc"

method = "bilinear"
periodic = False

ds_old = xr.open_dataset(file_old)
ds_ref = xr.open_dataset(file_ref)

dims = ds_old[var_name].dims
attrs = ds_old[var_name].attrs

dims_ref = ds_ref["sfc_temp"].dims

lat_ref = ds_ref[dims_ref[-2]].values
lon_ref = ds_ref[dims_ref[-1]].values

ds_new = xr.Dataset(
    {
        dims[-2]: ([dims[-2]], lat_ref, {"units": "degrees_north"}),
        dims[-1]: ([dims[-1]], lon_ref, {"units": "degrees_east"})
    }
)

regridder = xe.Regridder(ds_old, ds_new, method, periodic=periodic)

ds_new = regridder(ds_old)

dso = xr.Dataset()
dso[var_name] = ds_new[var_name].assign_attrs(attrs)

print(dso)

dso.to_netcdf(path=file_new, format="NETCDF4")