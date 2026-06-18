import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans

season = "jja"
resolution = "1"
experiment = "NO-URBAN"

if season == "djf":
    time_range_string = "201611300100-201703010000"
    time_start = "2016-12-01 00:00:00"
    time_end = "2017-02-28 23:00:00"
elif season == "jja":
    time_range_string = "201705310100-201709010000"
    time_start = "2017-06-01 00:00:00"
    time_end = "2017-08-31 23:00:00"

diri = f"/scratch/gb02/mr4682/GC26_sydney_wind/um2nc/SY_{season}/SY_{resolution}/{experiment}/"
diro = f"/scratch/gb02/mr4682/GC26_sydney_wind/wind_time_diurnal_max/SY_{season}/SY_{resolution}/{experiment}/"

file_u = f"{diri}wnd_ucmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"
file_v = f"{diri}wnd_vcmp-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"

fileo = f"{diro}time_diurnal_max-SY_SY_{resolution}_{experiment}-v1-{time_range_string}.nc"

level_height = xr.open_dataset(file_u)["level_height"]

u = xr.open_dataset(file_u)["wnd_ucmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))
v = xr.open_dataset(file_v)["wnd_vcmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))

if resolution == "1":
    u = u[..., :-1]
    v = v[..., :-1]

wind_speed = (u ** 2.0 + v ** 2.0) ** 0.5

print(wind_speed)

def max_hour_per_day(da):
    return da.time.dt.hour.isel(time=da.argmax('time'))

hour_of_max = wind_speed.groupby('time.date').map(max_hour_per_day)

print(hour_of_max)

time_diurnal_max = hour_of_max.mean(dim="date")
time_diurnal_max = time_diurnal_max.assign_attrs({"long_name": "Mean hour of daily maximum of wind speed"})

print(time_diurnal_max)

dso = xr.Dataset()
dso["time_diurnal_max"] = time_diurnal_max
dso["level_height"] = level_height

print(dso)

dso.to_netcdf(path=fileo, format="NETCDF4")