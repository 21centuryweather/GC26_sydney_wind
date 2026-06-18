import numpy as np
import pandas as pd
import xarray as xr

season = "jja"
resolution = "1"
experiment = "NO-URBAN"

if season == "djf":
    time_range_string = "201611300100-201703010000"
elif season == "jja":
    time_range_string = "201705310100-201709010000"

diri = f"/g/data/gb02/mjl561/um2nc/SY_{season}/SY_{resolution}/{experiment}/"
diro = f"/scratch/gb02/mr4682/GC26_sydney_wind/um2nc/SY_{season}/SY_{resolution}/{experiment}/"

file_ref = f"{diri}sfc_temp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"

file_u_old = f"{diri}wnd_ucmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"
file_v_old = f"{diri}wnd_vcmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"

file_u_new = f"{diro}wnd_ucmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"
file_v_new = f"{diro}wnd_vcmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"

u_old = xr.open_dataset(file_u_old)["wnd_ucmp"]
v_old = xr.open_dataset(file_v_old)["wnd_vcmp"]

ds_ref = xr.open_dataset(file_ref)

print(ds_ref)

dims_ref = ds_ref["sfc_temp"].dims

lat_ref = ds_ref[dims_ref[-2]]
lon_ref = ds_ref[dims_ref[-1]]

u_new = np.zeros((len(u_old[u_old.dims[0]].values), len(u_old[u_old.dims[1]].values), len(lat_ref.values), len(lon_ref.values)), dtype=u_old.values.dtype) * np.nan
v_new = np.zeros((len(v_old[v_old.dims[0]].values), len(v_old[v_old.dims[1]].values), len(lat_ref.values), len(lon_ref.values)), dtype=v_old.values.dtype) * np.nan

u_new[:, :, :, :-1] = (u_old[:, :, :, 1:].values + u_old[:, :, :, 0:-1].values) / 2.0
v_new = (v_old[:, :, 1:, :].values + v_old[:, :, 0:-1, :].values) / 2.0

u_new = xr.DataArray(u_new, dims=u_old.dims, coords={u_old.dims[0]: u_old[u_old.dims[0]], u_old.dims[1]: u_old[u_old.dims[1]], u_old.dims[2]: lat_ref, u_old.dims[3]: lon_ref}, attrs=u_old.attrs)
v_new = xr.DataArray(v_new, dims=v_old.dims, coords={v_old.dims[0]: v_old[v_old.dims[0]], v_old.dims[1]: v_old[v_old.dims[1]], v_old.dims[2]: lat_ref, v_old.dims[3]: lon_ref}, attrs=v_old.attrs)

dso_u = xr.Dataset()
dso_u["wnd_ucmp"] = u_new
dso_u["level_height"] = xr.open_dataset(file_u_old)["level_height"]

print(dso_u)

dso_u.to_netcdf(path=file_u_new, format="NETCDF4")

dso_v = xr.Dataset()
dso_v["wnd_vcmp"] = v_new
dso_v["level_height"] = xr.open_dataset(file_v_old)["level_height"]

print(dso_v)

dso_v.to_netcdf(path=file_v_new, format="NETCDF4")