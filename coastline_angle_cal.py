"""
Coastline Angle Calculation

This project uses two existing ACCESS-rAM high-resolution atmospheric model simulations over the Sydney region, run for two sets of three summer months (a summer and winter period), both with and without an urban representation. (data path: /g/data/gb02/mjl561/um2nc)
    - More detail. https://unsw-my.sharepoint.com/:w:/r/personal/z9901702_ad_unsw_edu_au/_layouts/15/Doc.aspx?sourcedoc=%7B760ec270-7524-43da-a870-02b02a74c8d5%7D&action=view&wdPid=572313e2

Based on the Python package "sea_breeze v1.1" from Andrew Brown
    - Method description: https://gmd.copernicus.org/articles/19/933/2026/
    - Code: https://zenodo.org/recorPythonds/17220916?preview_file=andrewbrown31%2Fsea_breeze-v1.1.zip

Modified by Chang Xu
"""

import sys
import os
import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin
    
import xarray as xr
import pandas as pd
from dask.distributed import Client
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.append('/home/561/cx5009/app/') # need change
client = Client()

from sea_breeze import (
load_model_data,
sea_breeze_funcs,
sea_breeze_filters
)

# =====================================================================
# Functions
# =====================================================================

def load_static(exp_season,exp_res,exp_id,lon_slice,lat_slice,chunks="auto"):

    """
    Load static fields for the ACCESS-rAM3 Sydney experiment, stored on the gb02 project.

    Parameters
    ----------
    exp_season : str
        Experiment Season. Must be one of 'SY_djf' or 'SY_jja'.
    exp_res : str
        Experiment Resolution. Must be one of 'SY_1', 'SY_5' or 'SY_11p1'.
    exp_id : str
        Experiment ID. Must be one of 'CTRL' or 'NO-URBAN'.
    lon_slice : slice or array-like
        Slice or indices to restrict longitude domain.
    lat_slice : slice or array-like
        Slice or indices to restrict latitude domain.
    chunks : str or dict, optional
        Chunking for xarray open_mfdataset (default is "auto").

    Returns
    -------
    orog : xarray.DataArray
        Orography field for the selected domain.
    lsm : xarray.DataArray
        Binary land-sea mask (1 for land, 0 for sea) for the selected domain.
    """

    orog = xr.open_dataset(f"/g/data/gb02/mjl561/um2nc/{exp_season}/{exp_res}/{exp_id}/topog-SY_{exp_res}_{exp_id}-v1.nc",chunks=chunks).sel(latitude=lat_slice,longitude=lon_slice)
    orog = orog.rename({
        "latitude": "lat",
        "longitude": "lon"
    })
    
    lsm = xr.open_dataset(f"/g/data/gb02/mjl561/um2nc/{exp_season}/{exp_res}/{exp_id}/lnd_mask-SY_{exp_res}_{exp_id}-v1.nc",chunks=chunks).sel(latitude=lat_slice,longitude=lon_slice)
    lsm = lsm.rename({
        "latitude": "lat",
        "longitude": "lon"
    })
    land_mask = xr.where(lsm.lnd_mask != 0, 1, 0)
    
    return orog.topog, land_mask

# =====================================================================
# Main
# =====================================================================
print("Identification Starting ......", flush=True)
print("=================== Setting params ===================", flush=True)
stats_output_path = "/g/data/up6/cx5009/hackathon/energy2026/"
      
#Lat lon and height bounds (Sydney, Australia). Height bounds chosen approximately as the typical maximum extent of the PBL
lat_slice = slice(-35.5,-32.269001)
lon_slice = slice(149.1505,153.1915)

exp_res = 'SY_5' # 'SY_1', 'SY_5', or 'SY_11p1'

print("=================== Coastline Angel Calculation ===================", flush=True)
#Load land sea mask
orog, lsm = load_static(exp_season,exp_res,exp_id,lon_slice,lat_slice)

#Compute coastline angles
angle_ds = load_model_data.get_coastline_angle_kernel(
    lsm,
    R=4,
    latlon_chunk_size=8,
    compute=True,
    smooth=False)

angle_ds.to_netcdf(f'/g/data/up6/cx5009/hackathon/energy2026/coast_angle_{exp_res}.nc')




