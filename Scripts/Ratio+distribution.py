#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:34:51 2025

@author: Jonna van Mourik
Make figure with MYD:ND ratio, together with drought distribution inlets
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append("/your/dir/")
from functions import MYD, ND, mask_MYD, mask_ND
import calendar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import re
import os
import pandas as pd
import cftime
from scipy.stats import ttest_1samp
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde


#Load in SPEI
dir = "/your/dir/"
models = ["CanESM5", "CESM2", "ACCESS-ESM1-5", "MIROC6", "EC-Earth3", "MPI-ESM1-2-LR"]


def convert_time(da):
    """Convert time coordinate to datetime64[ns] if stored as cftime.datetime."""
    if isinstance(da.time.values[0], cftime.datetime):
        da = da.assign_coords(time=pd.to_datetime(da.time.values.astype(str)))  # Convert to string first
    return da

# Load in masks
mask_AUS = xr.open_dataset(dir+"masks/1x1/mask_AUS_1x1.nc").Band1
mask_WEU = xr.open_dataset(dir+"masks/1x1/mask_WEU_1x1.nc").Band1
mask_IND = xr.open_dataset(dir+"masks/1x1/mask_IND_1x1.nc").Band1
mask_SA = xr.open_dataset(dir+"masks/1x1/mask_SA_1x1.nc").Band1
mask_CAL = xr.open_dataset(dir+"masks/1x1/mask_CAL_1x1.nc").Band1
mask_ARG = xr.open_dataset(dir+"masks/1x1/mask_ARG_1x1.nc").Band1

mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS]

#Cut-outs per region
lat_WEU = slice(42, 58) 
lon_WEU = slice(0, 17)
lat_IND = slice(16, 34)
lon_IND = slice(68, 94)
lat_AUS = slice(-40, -20)
lon_AUS = slice(135, 155)
lat_SA = slice(-33, -21)
lon_SA = slice(15, 31)
lat_ARG = slice(-45, -25)
lon_ARG = slice(-80, -55)
lat_CAL = slice(30, 44)
lon_CAL = slice(-124, -115)

lat_reg = [lat_CAL, lat_WEU, lat_IND, lat_ARG, lat_SA, lat_AUS]
lon_reg = [lon_CAL, lon_WEU, lon_IND, lon_ARG, lon_SA, lon_AUS]

reg = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
month_names = [calendar.month_abbr[m] for m in range(1,13)]

SPEI_grid = [xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/1x1/SPEI12_monthly_1950-2014_r*_"+str(model)+"_1x1.nc", combine="nested", concat_dim="member").sel(time=slice("1950", "2014")).__xarray_dataarray_variable__.resample(time="1MS").mean() for model in models]

def to_datetime64(ds):
    if isinstance(ds.indexes['time'], xr.CFTimeIndex):
        ds['time'] = ds.indexes['time'].to_datetimeindex()
    return ds

SPEI_grid_dt64 = [to_datetime64(ds) for ds in SPEI_grid]
SPEI_grid_all_models = xr.concat(SPEI_grid_dt64, dim="model", coords="minimal")

# Land cover, needed to mask ice, snow, and scarce vegetation
lc = [xr.open_dataset(dir+"masks/"+str(model)+"/1x1/landcover_"+str(model)+"_grid_1x1.nc").lccs_class for model in models]

def MYD_gridcell(spei_grid):
    # Create a mask where SPEI_AUS_grid is less than or equal to -1
    mask = spei_grid <= -1
    # Apply a rolling window of size 12 along the time dimension and check where the sum is 12
    rolling_sum = mask.rolling(time=12).sum()
    # Create a new mask where rolling_sum is greater than or equal to 12
    final_mask = rolling_sum >= 12
    # Convert the DataArray to a numpy array
    data_array = final_mask
    data_array_np = data_array.values
    # Iterate over each latitude and longitude
    for lat in range(data_array_np.shape[1]):
        for lon in range(data_array_np.shape[2]):
            # Iterate over each time step
            for t in range(data_array_np.shape[0]):
                # Check if the current value is True
                if data_array_np[t, lat, lon]:
                    # Set the 12 values before the True value to True
                    data_array_np[max(0, t-11):t, lat, lon] = True
    # Convert the modified numpy array back to a DataArray
    modified_data_array = xr.DataArray(data_array_np, coords=data_array.coords, dims=data_array.dims)
    # Print the modified DataArray
    #print(modified_data_array.sel(lon = 145, lat = -36)[-100:])
    # Convert the DataArray to a numpy array
    data_array_np = modified_data_array.values
    # Create a new array with 1 for True values and nan for False values
    new_mask_array = np.where(data_array_np, 1, np.nan)
    # Convert the new array back to a DataArray
    new_mask_data_array = xr.DataArray(new_mask_array, coords=data_array.coords, dims=data_array.dims)
    return new_mask_data_array

def number_of_droughts(SPEI_MYD_grid):
    da = SPEI_MYD_grid.where(SPEI_MYD_grid.notnull(), 0)
    drought_starts = da.diff("time")==1
    total_droughts = np.sum(drought_starts, axis=0)
    # Create a new xarray DataArray with the total number of droughts
    total_droughts_da = xr.DataArray(total_droughts, coords={'lat': da['lat'], 'lon': da['lon']}, dims=('lat', 'lon'))
    return total_droughts_da

all_SPEI_MYD_grid = []

for j in range(len(models)): 
    print("model=", j)
    SPEI_MYD_grid_list = []
    for i in range(10):
        SPEI_MYD_grid_list.append(MYD_gridcell(SPEI_grid[j].sel(member=i)).where(lc[j] != 220).where(lc[j] != 210).where(lc[j] != 200).where(lc[j] != 150))
    SPEI_MYD_grid = xr.concat(SPEI_MYD_grid_list, dim="member")
    all_SPEI_MYD_grid.append(convert_time(SPEI_MYD_grid.expand_dims(model=[models[j]])))

SPEI_MYD_grid_all_models = xr.concat(all_SPEI_MYD_grid, dim="model", coords='minimal')
SPEI_ND_grid_all_models = SPEI_grid_all_models.where((SPEI_MYD_grid_all_models!=1)&(SPEI_grid_all_models<=-1), np.nan)
SPEI_ND_grid_all_models = SPEI_ND_grid_all_models.where(SPEI_ND_grid_all_models.isnull(), 1)

n_droughts_all_models = xr.open_dataarray("/data/droughtTeam/n_droughts_all_models.nc")

#Calculate droughts in ERA
SPEI_grid_ERA = xr.open_dataarray(dir+"/SPEI/ERA/SPEI12_monthly_1950-2023_1x1.nc").sel(time=slice("1950", "2014"))

# Land cover, needed to mask ice, snow, and scarce vegetation
lc_ERA = xr.open_dataset(dir+"/SPEI/ERA/landcover_ERA_1x1.nc").lccs_class

SPEI_MYD_grid_ERA = MYD_gridcell(SPEI_grid_ERA).where(lc_ERA != 220).where(lc_ERA != 210).where(lc_ERA != 200).where(lc_ERA != 150) #150=sparce vegetation
SPEI_ND_grid_ERA = SPEI_grid_ERA.where((SPEI_MYD_grid_ERA!=1)&(SPEI_grid_ERA<=-1), np.nan).where(lc_ERA != 220).where(lc_ERA != 210).where(lc_ERA != 200).where(lc_ERA != 150)
SPEI_ND_grid_ERA = SPEI_ND_grid_ERA.where(SPEI_ND_grid_ERA.isnull(), 1)

n_droughts_ERA = number_of_droughts(SPEI_MYD_grid_ERA)
n_droughts_ERA = n_droughts_ERA.where(n_droughts_ERA != 0, float('nan'))

#Load in ratios
ratio_MMM = (SPEI_MYD_grid_all_models.count(dim="time")/SPEI_ND_grid_all_models.count(dim="time")).mean(dim=("member", "model")).persist()
ratio_ERA = (SPEI_MYD_grid_ERA.count(dim="time")/SPEI_ND_grid_ERA.count(dim="time")).persist()
ratio_MMM_std = (SPEI_MYD_grid_all_models.count(dim="time")/SPEI_ND_grid_all_models.count(dim="time")).std(dim=("member", "model")).persist()
ratio_diff = (ratio_MMM-ratio_ERA)/ratio_MMM_std

#%% Calculate what is needed for distribution
models = ["MIROC6", "MPI-ESM1-2-LR", "CanESM5", "ACCESS-ESM1-5", "EC-Earth3", "CESM2"]
startyear = "1950"
SPEI_reg_model = []
MYD_reg_model = []
ND_reg_model = []
for model in models:
    SPEI_AUS = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_AUS.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_WEU = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_WEU.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_CAL = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_CAL.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_IND = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_IND.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_SA = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_SA.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_ARG = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_ARG.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    
    #Australia
    MYD_AUS = MYD(SPEI_AUS, "AUS")
    ND_AUS = ND(SPEI_AUS, "AUS")
    
    #South Africa
    MYD_SA = MYD(SPEI_SA, "SA")
    ND_SA = ND(SPEI_SA, "SA")
    
    #California
    MYD_CAL = MYD(SPEI_CAL, "CAL")
    ND_CAL = ND(SPEI_CAL, "CAL")
          
    #Western Europe
    MYD_WEU = MYD(SPEI_WEU, "WEU")
    ND_WEU = ND(SPEI_WEU, "WEU")
    
    #Middle Argentina
    MYD_ARG = MYD(SPEI_ARG, "ARG")
    ND_ARG = ND(SPEI_ARG, "ARG")
    
    #India
    MYD_IND = MYD(SPEI_IND, "IND")
    ND_IND = ND(SPEI_IND, "IND")
    
    #Combine in lists
    SPEI_reg_model.append([SPEI_CAL, SPEI_WEU, SPEI_IND, SPEI_ARG, SPEI_SA, SPEI_AUS])
    MYD_reg_model.append([MYD_CAL, MYD_WEU, MYD_IND, MYD_ARG, MYD_SA, MYD_AUS])
    ND_reg_model.append([ND_CAL, ND_WEU, ND_IND, ND_ARG, ND_SA, ND_AUS])
    
#MYD
len_MYD_MIROC = [sum((sublist[2] for sublist in MYD_reg_model[0][5]), []), sum((sublist[2] for sublist in MYD_reg_model[0][4]), []), sum((sublist[2] for sublist in MYD_reg_model[0][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[0][2]), []), sum((sublist[2] for sublist in MYD_reg_model[0][1]), []), sum((sublist[2] for sublist in MYD_reg_model[0][0]), [])]
len_MYD_MPI = [sum((sublist[2] for sublist in MYD_reg_model[1][5]), []), sum((sublist[2] for sublist in MYD_reg_model[1][4]), []), sum((sublist[2] for sublist in MYD_reg_model[1][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[1][2]), []), sum((sublist[2] for sublist in MYD_reg_model[1][1]), []), sum((sublist[2] for sublist in MYD_reg_model[1][0]), [])]
len_MYD_CanESM = [sum((sublist[2] for sublist in MYD_reg_model[2][5]), []), sum((sublist[2] for sublist in MYD_reg_model[2][4]), []), sum((sublist[2] for sublist in MYD_reg_model[2][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[2][2]), []), sum((sublist[2] for sublist in MYD_reg_model[2][1]), []), sum((sublist[2] for sublist in MYD_reg_model[2][0]), [])]
len_MYD_ACCESS = [sum((sublist[2] for sublist in MYD_reg_model[3][5]), []), sum((sublist[2] for sublist in MYD_reg_model[3][4]), []), sum((sublist[2] for sublist in MYD_reg_model[3][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[3][2]), []), sum((sublist[2] for sublist in MYD_reg_model[3][1]), []), sum((sublist[2] for sublist in MYD_reg_model[3][0]), [])]
len_MYD_ECE = [sum((sublist[2] for sublist in MYD_reg_model[4][5]), []), sum((sublist[2] for sublist in MYD_reg_model[4][4]), []), sum((sublist[2] for sublist in MYD_reg_model[4][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[4][2]), []), sum((sublist[2] for sublist in MYD_reg_model[4][1]), []), sum((sublist[2] for sublist in MYD_reg_model[4][0]), [])]
len_MYD_CESM = [sum((sublist[2] for sublist in MYD_reg_model[5][5]), []), sum((sublist[2] for sublist in MYD_reg_model[5][4]), []), sum((sublist[2] for sublist in MYD_reg_model[5][3]), []),
           sum((sublist[2] for sublist in MYD_reg_model[5][2]), []), sum((sublist[2] for sublist in MYD_reg_model[5][1]), []), sum((sublist[2] for sublist in MYD_reg_model[5][0]), [])]
#ND
len_ND_MIROC = [sum((sublist[2] for sublist in ND_reg_model[0][5]), []), sum((sublist[2] for sublist in ND_reg_model[0][4]), []), sum((sublist[2] for sublist in ND_reg_model[0][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[0][2]), []), sum((sublist[2] for sublist in ND_reg_model[0][1]), []), sum((sublist[2] for sublist in ND_reg_model[0][0]), [])]
len_ND_MPI = [sum((sublist[2] for sublist in ND_reg_model[1][5]), []), sum((sublist[2] for sublist in ND_reg_model[1][4]), []), sum((sublist[2] for sublist in ND_reg_model[1][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[1][2]), []), sum((sublist[2] for sublist in ND_reg_model[1][1]), []), sum((sublist[2] for sublist in ND_reg_model[1][0]), [])]
len_ND_CanESM = [sum((sublist[2] for sublist in ND_reg_model[2][5]), []), sum((sublist[2] for sublist in ND_reg_model[2][4]), []), sum((sublist[2] for sublist in ND_reg_model[2][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[2][2]), []), sum((sublist[2] for sublist in ND_reg_model[2][1]), []), sum((sublist[2] for sublist in ND_reg_model[2][0]), [])]
len_ND_ACCESS = [sum((sublist[2] for sublist in ND_reg_model[3][5]), []), sum((sublist[2] for sublist in ND_reg_model[3][4]), []), sum((sublist[2] for sublist in ND_reg_model[3][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[3][2]), []), sum((sublist[2] for sublist in ND_reg_model[3][1]), []), sum((sublist[2] for sublist in ND_reg_model[3][0]), [])]
len_ND_ECE = [sum((sublist[2] for sublist in ND_reg_model[4][5]), []), sum((sublist[2] for sublist in ND_reg_model[4][4]), []), sum((sublist[2] for sublist in ND_reg_model[4][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[4][2]), []), sum((sublist[2] for sublist in ND_reg_model[4][1]), []), sum((sublist[2] for sublist in ND_reg_model[4][0]), [])]
len_ND_CESM = [sum((sublist[2] for sublist in ND_reg_model[5][5]), []), sum((sublist[2] for sublist in ND_reg_model[5][4]), []), sum((sublist[2] for sublist in ND_reg_model[5][3]), []),
           sum((sublist[2] for sublist in ND_reg_model[5][2]), []), sum((sublist[2] for sublist in ND_reg_model[5][1]), []), sum((sublist[2] for sublist in ND_reg_model[5][0]), [])]

labels_MYD = ["AUS", "SA", "ARG", "IND", "WEU", "CAL"]
labels_ND = [""]*len(labels_MYD)

len_all_MIROC = []
len_all_MPI = []
len_all_CanESM = []
len_all_ACCESS = []
len_all_ECE = []
len_all_CESM = []
len_all_MMM = []
for i in range(6):
    len_all_MMM.append([len_MYD_MIROC[i]+len_ND_MIROC[i]+len_MYD_MPI[i]+len_ND_MPI[i]
                        +len_MYD_CanESM[i]+len_ND_CanESM[i]+len_MYD_ACCESS[i]+len_ND_ACCESS[i]
                        +len_MYD_ECE[i]+len_ND_ECE[i]+len_MYD_CESM[i]+len_ND_CESM[i]])
    len_all_MIROC.append([len_MYD_MIROC[i]+len_ND_MIROC[i]])
    len_all_MPI.append([len_MYD_MPI[i]+len_ND_MPI[i]])
    len_all_CanESM.append([len_MYD_CanESM[i]+len_ND_CanESM[i]])
    len_all_ACCESS.append([len_MYD_ACCESS[i]+len_ND_ACCESS[i]])
    len_all_ECE.append([len_MYD_ECE[i]+len_ND_ECE[i]])
    len_all_CESM.append([len_MYD_CESM[i]+len_ND_CESM[i]])
colors = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]
len_all_list = [len_all_MIROC, len_all_MPI, len_all_CanESM, len_all_ACCESS, len_all_ECE, len_all_CESM]

#Also for ERA
SPEI_AUS_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_AUS.nc").sel(time=slice("1950", "2014"))
SPEI_ARG_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_ARG.nc").sel(time=slice("1950", "2014"))
SPEI_SA_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_SA.nc").sel(time=slice("1950", "2014"))
SPEI_WEU_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_WEU.nc").sel(time=slice("1950", "2014"))
SPEI_IND_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_IND.nc").sel(time=slice("1950", "2014"))
SPEI_CAL_ERA = xr.open_dataarray(dir+"SPEI/SPEI12_PM_monthly_1950_2014_0_5_degree_CAL.nc").sel(time=slice("1950", "2014"))

#Australia
MYD_AUS_ERA = MYD(SPEI_AUS_ERA, "AUS")
ND_AUS_ERA = ND(SPEI_AUS_ERA, "AUS")

#South Africa
MYD_SA_ERA = MYD(SPEI_SA_ERA, "SA")
ND_SA_ERA = ND(SPEI_SA_ERA, "SA")

#California
MYD_CAL_ERA = MYD(SPEI_CAL_ERA, "CAL")
ND_CAL_ERA = ND(SPEI_CAL_ERA, "CAL")
      
#Western Europe
MYD_WEU_ERA = MYD(SPEI_WEU_ERA, "WEU")
ND_WEU_ERA = ND(SPEI_WEU_ERA, "WEU")

#Middle Argentina
MYD_ARG_ERA = MYD(SPEI_ARG_ERA, "ARG")
ND_ARG_ERA = ND(SPEI_ARG_ERA, "ARG")

#India
MYD_IND_ERA = MYD(SPEI_IND_ERA, "IND")
ND_IND_ERA = ND(SPEI_IND_ERA, "IND")

MYD_ERA_reg = [MYD_CAL_ERA, MYD_WEU_ERA, MYD_IND_ERA, MYD_ARG_ERA, MYD_SA_ERA, MYD_AUS_ERA]
ND_ERA_reg = [ND_CAL_ERA, ND_WEU_ERA, ND_IND_ERA, ND_ARG_ERA, ND_SA_ERA, ND_AUS_ERA]

MYD_reg = [MYD_CAL, MYD_WEU, MYD_IND, MYD_ARG, MYD_SA, MYD_AUS]
ND_reg = [ND_CAL, ND_WEU, ND_IND, ND_ARG, ND_SA, ND_AUS]

len_MYD_ERA = [MYD_AUS_ERA[0][2], MYD_SA_ERA[0][2], MYD_ARG_ERA[0][2], MYD_IND_ERA[0][2], MYD_WEU_ERA[0][2], MYD_CAL_ERA[0][2]]
len_ND_ERA = [ND_AUS_ERA[0][2], ND_SA_ERA[0][2], ND_ARG_ERA[0][2], ND_IND_ERA[0][2], ND_WEU_ERA[0][2], ND_CAL_ERA[0][2]]

len_all_ERA = []
for i in range(6):
    len_all_ERA.append([MYD_ERA_reg[i][0][2]+ND_ERA_reg[i][0][2]])

x = np.linspace(0, 60, 1000)

#%% Version without inlets

#Plot
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
im = ratio_MMM.where(~lc_ERA.isin([220, 210, 200, 150])).plot(ax=ax, transform=ccrs.PlateCarree(), cmap="PRGn", vmin=0, vmax=2, cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=38, location="bottom", label="Ratio months in MYDs/NDs, MMM"))
cbar = im.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_xlabel("Ratio months in MYD:ND for MMM", fontsize=14)
pos = cbar.ax.get_position()       # get current position [x0, y0, width, height]
cbar.ax.set_position([pos.x0+0.01, pos.y0+0.03, pos.width-0.05, pos.height])
ax.coastlines()
ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)
ax.axis("off")
ax.set_title(" ")
#Colours for different regions
cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]


fig.savefig("world_map_ratio_v1.pdf", bbox_inches="tight")
fig.savefig("world_map_ratio_v1.jpg", bbox_inches="tight", dpi=1200)


#%% Make one version where all distributions are underneath each other
cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]

fig, axes = plt.subplots(6,1, figsize=(3,12), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    kde_all_models = []
    for j in range(len(models)):
        kde_model = gaussian_kde(len_all_list[j][5-i][0])
        kde_values = kde_model(x)
        area = np.trapz(kde_values, x)
        kde_values_norm = kde_values/area
        kde_all_models.append(kde_values_norm)
        #ax.plot(x, kde_values_norm, label=models[j], color=colors[j], alpha=0.5, linewidth=1)#linestyle="dotted")
    #ERA
    kde_ERA = gaussian_kde(len_all_ERA[i][0])
    kde_values_ERA = kde_ERA(x)
    area_ERA = np.trapz(kde_values_ERA, x)
    kde_values_norm_ERA = kde_values_ERA/area_ERA
    ax.plot(x, kde_values_norm_ERA, color="black", label="ERA5", linewidth=1)
    #MMM
    kde_MMM = gaussian_kde(len_all_MMM[5-i][0])
    kde_values_MMM = kde_MMM(x)
    area_MMM=np.trapz(kde_values_MMM, x)
    kde_values_norm_MMM = kde_values_MMM/area_MMM
    ax.plot(x, kde_values_norm_MMM, label="MMM", color="purple", linewidth=1)
    # Compute min and max for band
    kde_stack = np.vstack(kde_all_models)
    kde_min = np.min(kde_stack, axis=0)
    kde_max = np.max(kde_stack, axis=0)
    ax.fill_between(x, kde_min, kde_max, color="purple", alpha=0.3, label="Model range")

        
    ax.set_title(labels_MYD[5-i], fontsize=14, color=cb_color[i])
    #ax.text(0.95, 0.95, labels_MYD[5-i], color=cb_color[i], fontweight="bold", fontsize=14, ha="right", va="top", transform=ax_inset.transAxes)   # relative to axis coordinates

    ax.axvline(x=11.5, color='black', linewidth=0.5, linestyle="dashed")
    ax.set_ylim(0, 0.19)
    ax.set_xlim(0, 30)
    if i==5:
        ax.set_xlabel("Months in drought", fontsize=14)
fig.supylabel("Density", fontsize=14)
fig.subplots_adjust(left=0.30)

axes[0].legend(loc=[0.03, 1.3])
fig.savefig("distribution_months_drought_v1.pdf", bbox_inches="tight")
fig.savefig("distribution_months_drought_v1.jpg", bbox_inches="tight", dpi=1200)

#%% Difference between ERA5 and CMIP6 ratio
# Base colormap
cmap_diff = plt.get_cmap("BrBG_r")

# Sample 6 colors (3 negative, 3 positive), skip the center bin
colors_raw = [cmap_diff(i / 5) for i in range(6)]

# Construct new color list: insert white in the middle
# [−3, −2), [−2, −1), [−1, −0.5), [−0.5, 0.5), [0.5, 1), [1, 2), [2, 3)
colors_all = colors_raw[:3] + [(1, 1, 1, 1)] + [(1,1,1,1)] + colors_raw[3:]

# Create new colormap and boundary norm
diff_cmap = ListedColormap(colors_all)
bounds = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
norm = BoundaryNorm(bounds, ncolors=diff_cmap.N)

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
#Ratio of months
im2 = ratio_diff.plot(ax=ax, vmin=-3, vmax=3, transform=ccrs.PlateCarree(), cmap=diff_cmap, norm=norm, cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=38,location="bottom", label="Relative difference in ratio of MYD/ND between MMM and ERA5 in 1950-2014", extend="both"))
cbar2 = im2.colorbar
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.set_xlabel(r"Difference in ratio of MYD:ND between MMM and ERA5 in 1950-2014 ($\sigma$)", fontsize=14)
pos2 = cbar2.ax.get_position()       # get current position [x0, y0, width, height]
cbar2.ax.set_position([pos2.x0+0.01, pos2.y0+0.03, pos2.width-0.05, pos2.height])
ax.coastlines()
ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)
ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.set_title(" ")
ax.axis("off")
fig.savefig("world_map_ratio_diff_MMM_ERA5_v3.pdf", bbox_inches="tight")
fig.savefig("world_map_ratio_diff_MMM_ERA5_v3.jpg", bbox_inches="tight", dpi=1200)   
