import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans

season = "jja"
resolution = "1"
n_clusters = 5

if season == "djf":
    time_range_string = "201611300100-201703010000"
    time_start = "2016-12-01 00:00:00"
    time_end = "2017-02-28 23:00:00"
elif season == "jja":
    time_range_string = "201705310100-201709010000"
    time_start = "2017-06-01 00:00:00"
    time_end = "2017-08-31 23:00:00"

diri = f"/scratch/gb02/mr4682/GC26_sydney_wind/um2nc/SY_{season}/SY_{resolution}/"
diro = f"/scratch/gb02/mr4682/GC26_sydney_wind/wind_diurnal_vertical_profile_clustering/SY_{season}/SY_{resolution}/"

file_u_CTRL = f"{diri}CTRL/wnd_ucmp-SY_SY_{resolution}_CTRL-v1-{time_range_string}.nc"
file_v_CTRL = f"{diri}CTRL/wnd_vcmp-SY_SY_{resolution}_CTRL-v1-{time_range_string}.nc"

file_u_NOUR = f"{diri}NO-URBAN/wnd_ucmp-SY_SY_{resolution}_NO-URBAN-v1-{time_range_string}.nc"
file_v_NOUR = f"{diri}NO-URBAN/wnd_vcmp-SY_SY_{resolution}_NO-URBAN-v1-{time_range_string}.nc"

fileo = f"{diro}k-means_{n_clusters:d}_clusters-SY_SY_{resolution}_{time_range_string}.nc"

level_height = xr.open_dataset(file_u_CTRL)["level_height"]

u_CTRL = xr.open_dataset(file_u_CTRL)["wnd_ucmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))
v_CTRL = xr.open_dataset(file_v_CTRL)["wnd_vcmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))

u_NOUR = xr.open_dataset(file_u_NOUR)["wnd_ucmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))
v_NOUR = xr.open_dataset(file_v_NOUR)["wnd_vcmp"].sel(time=slice(time_start, time_end)).sel(latitude=slice(-35.5, -32.0), longitude=slice(149.0, 153.5))

if resolution == "1":
    u_CTRL = u_CTRL[..., :-1]
    v_CTRL = v_CTRL[..., :-1]
    u_NOUR = u_NOUR[..., :-1]
    v_NOUR = v_NOUR[..., :-1]

wind_speed_CTRL = (u_CTRL ** 2.0 + v_CTRL ** 2.0) ** 0.5
wind_speed_NOUR = (u_NOUR ** 2.0 + v_NOUR ** 2.0) ** 0.5

wind_speed_CTRL_hourly_mean = wind_speed_CTRL.groupby("time.hour").mean()
wind_speed_NOUR_hourly_mean = wind_speed_NOUR.groupby("time.hour").mean()

wind_speed_CTRL_hourly_mean = wind_speed_CTRL_hourly_mean.assign_attrs({"long_name": "Hourly mean of wind speed", "units": "m s**-1"})
wind_speed_NOUR_hourly_mean = wind_speed_NOUR_hourly_mean.assign_attrs({"long_name": "Hourly mean of wind speed", "units": "m s**-1"})

print(wind_speed_CTRL_hourly_mean)
print(wind_speed_NOUR_hourly_mean)

dim_0_len = len(wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[0]].values)
dim_1_len = len(wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[1]].values)
dim_2_len = len(wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[2]].values)
dim_3_len = len(wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[3]].values)

print(f"[{dim_0_len}, {dim_1_len}, {dim_2_len}, {dim_3_len}]")

input_matrix_CTRL = wind_speed_CTRL_hourly_mean.values.reshape((dim_0_len * dim_1_len, dim_2_len, dim_3_len))
input_matrix_CTRL = wind_speed_CTRL_hourly_mean.values.reshape((dim_0_len * dim_1_len, dim_2_len * dim_3_len))
input_matrix_CTRL = input_matrix_CTRL.T

print(input_matrix_CTRL.shape)

input_matrix_NOUR = wind_speed_NOUR_hourly_mean.values.reshape((dim_0_len * dim_1_len, dim_2_len, dim_3_len))
input_matrix_NOUR = wind_speed_NOUR_hourly_mean.values.reshape((dim_0_len * dim_1_len, dim_2_len * dim_3_len))
input_matrix_NOUR = input_matrix_NOUR.T

print(input_matrix_NOUR.shape)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(input_matrix_CTRL)

patterns = kmeans.cluster_centers_
patterns = patterns.reshape((patterns.shape[0], dim_0_len, dim_1_len))

clusters = np.arange(1, patterns.shape[0] + 1, 1, dtype=int)
clusters = xr.DataArray(clusters, dims=["cluster"], coords={"cluster": clusters}, attrs={"long_name": "Cluster"})

patterns = xr.DataArray(patterns, dims=["cluster", wind_speed_CTRL_hourly_mean.dims[0], wind_speed_CTRL_hourly_mean.dims[1]], coords={"cluster": clusters, wind_speed_CTRL_hourly_mean.dims[0]: wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[0]], wind_speed_CTRL_hourly_mean.dims[1]: wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[1]]}, attrs={"long_name": "k-means clustering patterns"})

labels_CTRL = kmeans.labels_ + 1
labels_CTRL = labels_CTRL.reshape((dim_2_len, dim_3_len))
labels_CTRL = xr.DataArray(labels_CTRL, dims=[wind_speed_CTRL_hourly_mean.dims[2], wind_speed_CTRL_hourly_mean.dims[3]], coords={wind_speed_CTRL_hourly_mean.dims[2]: wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[2]], wind_speed_CTRL_hourly_mean.dims[3]: wind_speed_CTRL_hourly_mean[wind_speed_CTRL_hourly_mean.dims[3]]}, attrs={"long_name": "k-means clustering labels (CTRL)"})

labels_NOUR = kmeans.predict(input_matrix_NOUR) + 1
labels_NOUR = labels_NOUR.reshape((dim_2_len, dim_3_len))
labels_NOUR = xr.DataArray(labels_NOUR, dims=[wind_speed_NOUR_hourly_mean.dims[2], wind_speed_NOUR_hourly_mean.dims[3]], coords={wind_speed_NOUR_hourly_mean.dims[2]: wind_speed_NOUR_hourly_mean[wind_speed_NOUR_hourly_mean.dims[2]], wind_speed_NOUR_hourly_mean.dims[3]: wind_speed_NOUR_hourly_mean[wind_speed_NOUR_hourly_mean.dims[3]]}, attrs={"long_name": "k-means clustering labels (NO-URBAN)"})

dso = xr.Dataset()
dso["pattern"] = patterns
dso["label_CTRL"] = labels_CTRL
dso["label_NO-URBAN"] = labels_NOUR
dso["level_height"] = level_height

print(dso)

dso.to_netcdf(path=fileo, format="NETCDF4")