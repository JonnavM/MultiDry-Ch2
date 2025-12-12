#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:17:46 2025

@author: Jonna van Mourik
Climatological plots for precipitation, PET and the number of MYDs for all models together
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

#Load in SPEI
dir = "/your/dir/"
models = ["CanESM5", "CESM2", "ACCESS-ESM1-5", "MIROC6", "EC-Earth3", "MPI-ESM1-2-LR"]
pr_list = []
pet_list = []

def convert_time(da):
    """Convert time coordinate to datetime64[ns] if stored as cftime.datetime."""
    if isinstance(da.time.values[0], cftime.datetime):
        da = da.assign_coords(time=pd.to_datetime(da.time.values.astype(str)))  # Convert to string first
    return da

#Load in PET and pr. both in mm/day
for model in models:
    #For EC-Earth3
    if model =="EC-Earth3":
        print(model)
        grid = "gr"
        common_lat = xr.open_dataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r1i1p1f1_"+str(grid)+"_1850-2014_1x1.nc")['lat'].values
        def preprocess(ds):
            return ds.reindex(lat=common_lat, method='nearest')  # Align to the common grid
        pr = xr.open_mfdataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r*i1p1f1_"+str(grid)+"_1850-2014_1x1.nc", preprocess=preprocess, combine='nested', concat_dim='member').pr.sel(time=slice("1950", "2014"))*3600*24
        pet = xr.open_mfdataset(dir+"PET/"+str(model)+"/1x1/pm_fao56_r*_1850-2014_monthly_"+str(model)+"_hist_1x1.nc", preprocess=preprocess, combine="nested", concat_dim="member").PM_FAO_56.sel(time=slice("1950", "2014"))
    else:
        print(model)
        grid = "gn"
        pr = xr.open_mfdataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r*i1p1f1_"+str(grid)+"_1850-2014_1x1.nc", combine='nested', concat_dim='member').pr.sel(time=slice("1950", "2014"))*3600*24
        pet = xr.open_mfdataset(dir+"PET/"+str(model)+"/1x1/pm_fao56_r*_1850-2014_monthly_"+str(model)+"_hist_1x1.nc", combine='nested', concat_dim='member').PM_FAO_56.sel(time=slice("1950", "2014"))
    #Change units to mm/month
    days_in_month = pr.time.dt.days_in_month
    pr_month = (pr*days_in_month).resample(time="1MS").mean()
    days_in_month = pet.time.dt.days_in_month
    pet_month = (pet*days_in_month).resample(time="1MS").mean()
    #pet_month["lat"]=pr_month["lat"]
    pr_month.attrs['units'] = 'mm/month'
    pet_month.attrs['units'] = 'mm/month'
    pr_list.append(convert_time(pr_month))
    pet_list.append(convert_time(pet_month))
    print(pr_month.time)
    
pr_month_all = xr.concat(pr_list, dim="model", coords="minimal").load()
pet_month_all = xr.concat(pet_list, dim="model", coords="minimal").load()
pr_month_all["model"]=models
pet_month_all["model"]=models

#Load in PET and pr for ERA
pr_ERA = (xr.open_dataarray(dir+"ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc")*1000).sel(time=slice("1950", "2014")) #Units: mm/day
pet_ERA = xr.open_dataset(dir+"PET/PenmanMonteith/ERA5/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56.sel(time=slice("1950", "2014")) #Units: mm/day
pet_ERA["time"]=pr_ERA["time"]
#Change units to mm/month
days_in_month_ERA = pr_ERA.time.dt.days_in_month
pr_month_ERA = pr_ERA*days_in_month_ERA
pet_month_ERA = pet_ERA*days_in_month_ERA
pr_month_ERA.attrs['units'] = 'mm/month'
pet_month_ERA.attrs['units'] = 'mm/month'
pr_month_ERA = pr_month_ERA.interp(lat=pr_month_all.lat, lon=pr_month_all.lon)
pet_month_ERA = pet_month_ERA.interp(lat=pet_month_all.lat, lon=pet_month_all.lon)

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

reg = ["CAL", "  WEU", "IND", "ARG", "SA", "AUS"]
month_names = [calendar.month_abbr[m] for m in range(1,13)]

#%% Definitions
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
all_n_droughts = []

for j in range(len(models)): 
    print("model=", j)
    SPEI_MYD_grid_list = []
    for i in range(10):
        SPEI_MYD_grid_list.append(MYD_gridcell(SPEI_grid[j].sel(member=i)).where(~lc[j].isin([220, 210, 200, 150])))
    SPEI_MYD_grid = xr.concat(SPEI_MYD_grid_list, dim="member")
    all_SPEI_MYD_grid.append(convert_time(SPEI_MYD_grid.expand_dims(model=[model])))

    #Save number of droughts
    """
    n_droughts_list = []
    for i in range(10):
        print("member=", i)
        n_droughts = number_of_droughts(SPEI_MYD_grid.sel(member=i))
        n_droughts = n_droughts.where(n_droughts != 0, float('nan'))
        n_droughts_list.append(n_droughts)
    n_droughts = xr.concat(n_droughts_list, dim="member")

    # Store results in the model list
    all_n_droughts.append(n_droughts.expand_dims(model=[model]))
    """
SPEI_MYD_grid_all_models = xr.concat(all_SPEI_MYD_grid, dim="model", coords='minimal')
SPEI_ND_grid_all_models = SPEI_grid_all_models.where((SPEI_MYD_grid_all_models!=1)&(SPEI_grid_all_models<=-1), np.nan)
SPEI_ND_grid_all_models = SPEI_ND_grid_all_models.where(SPEI_ND_grid_all_models.isnull(), 1)
#n_droughts_all_models = xr.concat(all_n_droughts, dim="model", coords="minimal")
#n_droughts_all_models
#n_droughts_all_models.to_netcdf("/data/droughtTeam/n_droughts_all_models.nc")
n_droughts_all_models = xr.open_dataarray("/data/droughtTeam/n_droughts_all_models.nc")
n_droughts_all_models["model"] = models

#Calculate droughts in ERA
SPEI_grid_ERA = xr.open_dataarray(dir+"/SPEI/ERA/SPEI12_monthly_1950-2023_1x1.nc").sel(time=slice("1950", "2014"))

# Land cover, needed to mask ice, snow, and scarce vegetation
lc_ERA = xr.open_dataset(dir+"/SPEI/ERA/landcover_ERA_1x1.nc").lccs_class

SPEI_MYD_grid_ERA = MYD_gridcell(SPEI_grid_ERA).where(~lc_ERA.isin([220, 210, 200, 150])) #150=sparce vegetation
SPEI_ND_grid_ERA = SPEI_grid_ERA.where((SPEI_MYD_grid_ERA!=1)&(SPEI_grid_ERA<=-1), np.nan).where(~lc_ERA.isin([220, 210, 200, 150]))
SPEI_ND_grid_ERA = SPEI_ND_grid_ERA.where(SPEI_ND_grid_ERA.isnull(), 1)

n_droughts_ERA = number_of_droughts(SPEI_MYD_grid_ERA)
n_droughts_ERA = n_droughts_ERA.where(n_droughts_ERA != 0, float('nan'))

#%% Calculate internal variability
std = n_droughts_all_models.std(dim=("member", "model"))
members = np.arange(0,10)

n_dm = n_droughts_all_models.where(~n_droughts_all_models.isnull().all(dim=("model", "member")))
mean_all = n_dm.mean(dim=("model", "member"))        
diffs = abs(n_dm - mean_all)
diffs_zeroed = diffs.fillna(0)
internal_var = diffs_zeroed.mean(dim=("model", "member"))/std
internal_var = internal_var.where(~n_droughts_all_models.isnull().all(dim=("model","member")))

diff_ERA = (abs(n_droughts_all_models.mean(dim=("member", "model"))-n_droughts_ERA))/std
bias_ERA = diff_ERA-internal_var

#%% Plot map of the world
#Make own colormap
cmap_continuous = plt.get_cmap("magma_r")  

#Plot
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
n_droughts_all_models.mean(dim=("member", "model")).where(~lc_ERA.isin([220, 210, 200, 150])).plot(transform=ccrs.PlateCarree(), cmap=cmap_continuous, vmin=1.5, vmax=6.5, extend="both", cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=40, location="bottom", label="Number of MYDs in MMM between 1950-2014", ticks=[1,2,3,4,5,6,7,8]))
ax.coastlines()
ax.set_extent([-182, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)

ax.axis("off")

cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]
month_letter = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for i, region_name in enumerate(reg):
    if region_name == "IND": #IND, AUS, WEU, SA, SSA, CAL
        reg_lat = slice(22, 32)
        reg_lon = slice(75, 86.5)
        left = 0.82 #Fraction of figure, 0.84 for PlateCarree
        bottom = 0.48 #Fraction of figure, 0.36 for PlateCarree
        inset_center_x = 145 #End of arrow, 136 for Platecaree
        inset_center_y = 25 #Start of arrow, 0.25 for PlateCarree
        box_center_lon = reg_lon.stop #-13 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "AUS":
        reg_lat = slice(-40, -22)
        reg_lon = slice(137, 155)
        left = 0.6 #0.63 for PlateCarree
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = 116 #97 for PlateCarree
        inset_center_y = -41 #-40 for PlateCarree
        box_center_lon = reg_lon.start #-6 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 3
    elif region_name == "  WEU":
        reg_lat = slice(45, 55)
        reg_lon = slice(0.5, 13)
        left = 0.29 #0.37 for PlateCarree
        bottom = 0.51 #0.445 for PlateCarree
        inset_center_x = -24 #-30 for PlateCarree
        inset_center_y = 43 #-47 for PlateCarree
        box_center_lon = reg_lon.start #-9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "SA":
        reg_lat = slice(-33, -21)
        reg_lon = slice(16, 31)
        left = 0.362
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = 10 #-5 for PlateCarree
        inset_center_y = -32 #-28 for PlateCarree
        box_center_lon = reg_lon.start +4# - 9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 5
    elif region_name == "ARG":
        reg_lat = slice(-43, -26)
        #reg_lon = slice(280, 305)
        reg_lon = slice(-80, -55)
        left = 0.14 #0.2 for PlateCarree
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = -85 #-105 for PlateCarree
        inset_center_y = -35 #-35 for PlateCarree
        box_center_lon = reg_lon.start + 9 #-1 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "CAL":
        reg_lat = slice(32, 41.5)
        #reg_lon = slice(236, 245)
        reg_lon = slice(-124, -115)
        left = 0.035 #0.11 for PlateCarree
        bottom = 0.53 #0.3.. for PlateCarree
        inset_center_x = -130 #-149 for PlateCarree
        inset_center_y = 36 #36 for PlateCarree
        box_center_lon = reg_lon.start + 3 #-9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    if region_name == "IND":
        left, bottom, width, height = [left, bottom, 0.1, 0.2] #0.08, 0.16 for PlateCarree
    else: 
        left, bottom, width, height = [left, bottom, 0.1, 0.1] #0.08, 0.08 for PlateCarree
    fig.patch.set_facecolor('white')
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.set_facecolor('white')
    
    pr_reg = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
    pr_reg_std = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time", "model"))
    pet_reg = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
    pet_reg_std = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time", "model"))

    ax_inset.bar(pr_reg.month, pr_reg, color="tab:blue", yerr=pr_reg_std, label="Pr")
    ax_inset.set_xticks(pr_reg.month, month_letter, fontsize=8)

    if region_name == "IND":
        ax_inset.set_ylim(0,440)
    else:
        ax_inset.set_ylim(0, 220)
    ax_inset.tick_params(axis='y', labelsize=8)
    pet_reg.plot(ax=ax_inset, color="red", label="PET")
    ax_inset.fill_between(x=pet_reg.month, y1=pet_reg-pet_reg_std, y2=pet_reg+pet_reg_std, color="red", alpha=0.2)
    ax_inset.set_title(reg[i], color=cb_color[i], fontweight="bold")
    
    # Calculate total annual precipitation and PET
    total_pr_ann = pr_reg.sum().values
    total_pet_ann = pet_reg.sum().values
    
    # Add text with total annual values in the upper right corner
    if region_name == "ARG":
        ax_inset.set_ylabel("[mm/month]", fontsize=8)
        ax_inset.text(12.7, 190, f"PET:{total_pet_ann:.0f} mm/yr", color="red", fontsize=8, horizontalalignment="right")
        ax_inset.text(12.7, 160, f"PR:{total_pr_ann:.0f} mm/yr", color="tab:blue", fontsize=8, horizontalalignment="right")
    elif region_name == "IND":
        ax_inset.text(12.4, 410, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
        ax_inset.text(12.4, 380, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
    elif region_name == "SA" or region_name == "AUS":
        ax_inset.set_ylabel(" ")
        ax_inset.text(6.5, 190, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='center')
        ax_inset.text(6.5, 160, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='center')
    else:
        ax_inset.set_ylabel(" ")
        ax_inset.text(12.4, 190, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
        ax_inset.text(12.4, 160, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')

    ax_inset.set_xlabel(" ")
    # Draw an arrow pointing towards the inset plot
    ax.annotate("", xy=(inset_center_x, inset_center_y), xytext=(box_center_lon, box_center_lat),
                    arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2), fontsize=12, transform=ccrs.PlateCarree())
    
fig.savefig("world_map_MYDS+climate_1950-2014_allModels_1x1_v4.pdf", bbox_inches="tight")
fig.savefig("world_map_MYDS+climate_1950-2014_allModels_1x1_v4.jpg", bbox_inches="tight", dpi=1200)

#%% And a version where the difference is divided by the std
print("Plot difference MMM and ERA5")
#std = n_droughts_all_models.mean("member").std("model")
std = n_droughts_all_models.std(dim=("member", "model"))
diff = n_droughts_all_models.mean(dim=("member", "model"))-n_droughts_ERA

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


#Plot
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
(diff/std).plot(transform=ccrs.PlateCarree(), cmap=diff_cmap, norm=norm, cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=40, location="bottom", label=r"Diffference in number of MYDs between MMM and ERA5 in 1950-2014 ($\sigma$)"))
ax.coastlines()
ax.set_extent([-182, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)
sig_mask = abs(diff/std)>internal_var
sig_mask_int = sig_mask.astype(int)

cs = ax.contourf(
    sig_mask_int.lon,
    sig_mask_int.lat,
    sig_mask_int,
    levels=[0.5, 1.5],      # 1 = significant
    hatches=['////'],        # hatch pattern
    colors='none',           # no fill
    transform=ccrs.PlateCarree()
)

ax.axis("off")

cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]
month_letter = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for i, region_name in enumerate(reg):
    if region_name == "IND": #IND, AUS, WEU, SA, SSA, CAL
        reg_lat = slice(22, 32)
        reg_lon = slice(75, 86.5)
        left = 0.82 #Fraction of figure, 0.84 for PlateCarree
        bottom = 0.48 #Fraction of figure, 0.36 for PlateCarree
        inset_center_x = 145 #End of arrow, 136 for Platecaree
        inset_center_y = 25 #Start of arrow, 0.25 for PlateCarree
        box_center_lon = reg_lon.stop #-13 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "AUS":
        reg_lat = slice(-40, -22)
        reg_lon = slice(137, 155)
        left = 0.6 #0.63 for PlateCarree
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = 116 #97 for PlateCarree
        inset_center_y = -41 #-40 for PlateCarree
        box_center_lon = reg_lon.start #-6 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 3
    elif region_name == "  WEU":
        reg_lat = slice(45, 55)
        reg_lon = slice(0.5, 13)
        left = 0.29 #0.37 for PlateCarree
        bottom = 0.51 #0.445 for PlateCarree
        inset_center_x = -24 #-30 for PlateCarree
        inset_center_y = 43 #-47 for PlateCarree
        box_center_lon = reg_lon.start #-9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "SA":
        reg_lat = slice(-33, -21)
        reg_lon = slice(16, 31)
        left = 0.362
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = 10 #-5 for PlateCarree
        inset_center_y = -32 #-28 for PlateCarree
        box_center_lon = reg_lon.start +4# - 9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 5
    elif region_name == "ARG":
        reg_lat = slice(-43, -26)
        #reg_lon = slice(280, 305)
        reg_lon = slice(-80, -55)
        left = 0.14 #0.2 for PlateCarree
        bottom = 0.3 #0.28 for PlateCarree
        inset_center_x = -85 #-105 for PlateCarree
        inset_center_y = -35 #-35 for PlateCarree
        box_center_lon = reg_lon.start + 9 #-1 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "CAL":
        reg_lat = slice(32, 41.5)
        #reg_lon = slice(236, 245)
        reg_lon = slice(-124, -115)
        left = 0.035 #0.11 for PlateCarree
        bottom = 0.53 #0.3.. for PlateCarree
        inset_center_x = -130 #-149 for PlateCarree
        inset_center_y = 36 #36 for PlateCarree
        box_center_lon = reg_lon.start + 3 #-9 for PlateCarree
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    if region_name == "IND":
        left, bottom, width, height = [left, bottom, 0.1, 0.2] #0.08, 0.16 for PlateCarree
    else: 
        left, bottom, width, height = [left, bottom, 0.1, 0.1] #0.08, 0.08 for PlateCarree
    fig.patch.set_facecolor('white')
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.set_facecolor('white')
    
    #Calculate mean and std for models
    pr_reg = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
    pr_reg_std = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time", "model"))
    pet_reg = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
    pet_reg_std = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time", "model"))
    
    #Do the same for ERA
    pr_reg_ERA = pr_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
    pr_reg_ERA_std = pr_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")
    pet_reg_ERA = pet_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
    pet_reg_ERA_std = pet_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")
    
    #Calculate differnces and total std
    pr_diff = pr_reg - pr_reg_ERA
    pr_total_std = np.sqrt(pr_reg_std**2 + pr_reg_ERA_std**2)
    pet_diff = pet_reg - pet_reg_ERA
    pet_total_std = np.sqrt(pet_reg_std**2 + pet_reg_ERA_std**2)

    #Plot in figure
    ax_inset.bar(pr_diff.month, pr_diff, color="tab:blue", yerr=pr_total_std, label="Pr")
    ax_inset.set_xticks(pr_reg.month, month_letter, fontsize=8)

    if region_name == "IND":
        ax_inset.set_ylim(-400,200) #Was 0,440
    else:
        ax_inset.set_ylim(-150, 150) #was 0,220
    ax_inset.tick_params(axis='y', labelsize=8)
    pet_diff.plot(ax=ax_inset, color="red", label="PET")
    ax_inset.fill_between(x=pet_diff.month, y1=pet_diff-pet_total_std, y2=pet_diff+pet_total_std, color="red", alpha=0.2)
    ax_inset.set_title(reg[i], color=cb_color[i], fontweight="bold")
    
    # Calculate total annual precipitation and PET
    total_pr_ann = pr_diff.sum().values
    total_pet_ann = pet_diff.sum().values
    
    # Add text with total annual values in the upper right corner
    if region_name == "ARG":
        ax_inset.set_ylabel("[mm/month]", fontsize=8)
        ax_inset.text(12.7, -100, f"PET:{total_pet_ann:.0f} mm/yr", color="red", fontsize=8, horizontalalignment="right")
        ax_inset.text(12.7, -130, f"PR:{total_pr_ann:.0f} mm/yr", color="tab:blue", fontsize=8, horizontalalignment="right")
    elif region_name == "IND":
        ax_inset.text(12.4, -350, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
        ax_inset.text(12.4, -380, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
    elif region_name == "WEU":
        ax_inset.set_ylabel(" ")
        ax_inset.text(6.5, 110, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='center')
        ax_inset.text(6.5, 80, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='center')
    else:
        ax_inset.set_ylabel(" ")
        ax_inset.text(12.4, -100, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
        ax_inset.text(12.4, -130, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
    
    ax_inset.set_xlabel(" ")
    # Draw an arrow pointing towards the inset plot
    ax.annotate("", xy=(inset_center_x, inset_center_y), xytext=(box_center_lon, box_center_lat),
                    arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2), fontsize=12, transform=ccrs.PlateCarree())
    
fig.savefig("world_map_MYDS+climate_1950-2014_allModels(std-models+members)_diffSTD_sign_v1.pdf", bbox_inches="tight")
fig.savefig("world_map_MYDS+climate_1950-2014_allModels(std-models+members)_diffSTD_sign_v1.jpg", bbox_inches="tight", dpi=1200)
    
#%% Make one version where we test for the interannual and intramodel values. Plot each model against the MMM
#First by plotting the difference of each model against ERA5
print("Plot difference model and MMM")
#std = n_droughts_all_models.mean("member").std("model")
std = n_droughts_all_models.std(dim=("member", "model"))

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

for model in models:
    diff = n_droughts_all_models.sel(model=model).mean("member") - n_droughts_all_models.mean(dim=("model", "member"))

    #Plot
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
    (diff/std).plot(transform=ccrs.PlateCarree(), cmap=diff_cmap, norm=norm, cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=40, location="bottom", label=r"Diffference in number of MYDs between model and MMM in 1950-2014 ($\sigma$)"))
    ax.coastlines()
    ax.set_extent([-182, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
    ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)
    ax.axis("off")
    
    cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]
    month_letter = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    
    for i, region_name in enumerate(reg):
        if region_name == "IND": #IND, AUS, WEU, SA, SSA, CAL
            reg_lat = slice(22, 32)
            reg_lon = slice(75, 86.5)
            left = 0.82 #Fraction of figure, 0.84 for PlateCarree
            bottom = 0.48 #Fraction of figure, 0.36 for PlateCarree
            inset_center_x = 145 #End of arrow, 136 for Platecaree
            inset_center_y = 25 #Start of arrow, 0.25 for PlateCarree
            box_center_lon = reg_lon.stop #-13 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        elif region_name == "AUS":
            reg_lat = slice(-40, -22)
            reg_lon = slice(137, 155)
            left = 0.6 #0.63 for PlateCarree
            bottom = 0.3 #0.28 for PlateCarree
            inset_center_x = 116 #97 for PlateCarree
            inset_center_y = -41 #-40 for PlateCarree
            box_center_lon = reg_lon.start #-6 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 3
        elif region_name == "  WEU":
            reg_lat = slice(45, 55)
            reg_lon = slice(0.5, 13)
            left = 0.29 #0.37 for PlateCarree
            bottom = 0.51 #0.445 for PlateCarree
            inset_center_x = -24 #-30 for PlateCarree
            inset_center_y = 43 #-47 for PlateCarree
            box_center_lon = reg_lon.start #-9 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        elif region_name == "SA":
            reg_lat = slice(-33, -21)
            reg_lon = slice(16, 31)
            left = 0.362
            bottom = 0.3 #0.28 for PlateCarree
            inset_center_x = 10 #-5 for PlateCarree
            inset_center_y = -32 #-28 for PlateCarree
            box_center_lon = reg_lon.start +4# - 9 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 5
        elif region_name == "ARG":
            reg_lat = slice(-43, -26)
            #reg_lon = slice(280, 305)
            reg_lon = slice(-80, -55)
            left = 0.14 #0.2 for PlateCarree
            bottom = 0.3 #0.28 for PlateCarree
            inset_center_x = -85 #-105 for PlateCarree
            inset_center_y = -35 #-35 for PlateCarree
            box_center_lon = reg_lon.start + 9 #-1 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        elif region_name == "CAL":
            reg_lat = slice(32, 41.5)
            #reg_lon = slice(236, 245)
            reg_lon = slice(-124, -115)
            left = 0.035 #0.11 for PlateCarree
            bottom = 0.53 #0.3.. for PlateCarree
            inset_center_x = -130 #-149 for PlateCarree
            inset_center_y = 36 #36 for PlateCarree
            box_center_lon = reg_lon.start + 3 #-9 for PlateCarree
            box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        if region_name == "IND":
            left, bottom, width, height = [left, bottom, 0.1, 0.2] #0.08, 0.16 for PlateCarree
        else: 
            left, bottom, width, height = [left, bottom, 0.1, 0.1] #0.08, 0.08 for PlateCarree
        fig.patch.set_facecolor('white')
        ax_inset = fig.add_axes([left, bottom, width, height])
        ax_inset.set_facecolor('white')
        
        #Calculate mean and std for models
        pr_reg = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i], model=model).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").mean("time")
        pr_reg_std = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i], model=model).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time"))
        pet_reg = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i], model=model).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").mean("time")
        pet_reg_std = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i], model=model).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member")).groupby("time.month").std(dim=("time"))
        
        #Do the same for ERA of MMM depending on which we need
        pr_reg_ERA = pr_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
        pr_reg_ERA_std = pr_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")
        pet_reg_ERA = pet_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
        pet_reg_ERA_std = pet_month_ERA.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")
        #MMM
        pr_reg_MMM = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
        pr_reg_MMM_std = pr_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").std("time")
        pet_reg_MMM = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").mean("time")
        pet_reg_MMM_std = pet_month_all.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon", "member", "model")).groupby("time.month").std("time")
        
        #Calculate differnces and total std
        pr_diff = pr_reg - pr_reg_MMM
        pr_total_std = np.sqrt(pr_reg_std**2 + pr_reg_MMM_std**2)
        pet_diff = pet_reg - pet_reg_MMM
        pet_total_std = np.sqrt(pet_reg_std**2 + pet_reg_MMM_std**2)
    
        #Plot in figure
        ax_inset.bar(pr_diff.month, pr_diff, color="tab:blue", yerr=pr_total_std, label="Pr")
        ax_inset.set_xticks(pr_reg.month, month_letter, fontsize=8)
    
        if region_name == "IND":
            ax_inset.set_ylim(-400,200) #Was 0,440
        else:
            ax_inset.set_ylim(-150, 150) #was 0,220
        ax_inset.tick_params(axis='y', labelsize=8)
        pet_diff.plot(ax=ax_inset, color="red", label="PET")
        ax_inset.fill_between(x=pet_diff.month, y1=pet_diff-pet_total_std, y2=pet_diff+pet_total_std, color="red", alpha=0.2)
        ax_inset.set_title(reg[i], color=cb_color[i], fontweight="bold")
        
        # Calculate total annual precipitation and PET
        total_pr_ann = pr_diff.sum().values
        total_pet_ann = pet_diff.sum().values
        
        # Add text with total annual values in the upper right corner
        if region_name == "ARG":
            ax_inset.set_ylabel("[mm/month]", fontsize=8)
            ax_inset.text(12.7, -100, f"PET:{total_pet_ann:.0f} mm/yr", color="red", fontsize=8, horizontalalignment="right")
            ax_inset.text(12.7, -130, f"PR:{total_pr_ann:.0f} mm/yr", color="tab:blue", fontsize=8, horizontalalignment="right")
        elif region_name == "IND":
            ax_inset.text(12.4, -350, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
            ax_inset.text(12.4, -380, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
        elif region_name == "WEU":
            ax_inset.set_ylabel(" ")
            ax_inset.text(6.5, 110, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='center')
            ax_inset.text(6.5, 80, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='center')
        else:
            ax_inset.set_ylabel(" ")
            ax_inset.text(12.4, -100, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')
            ax_inset.text(12.4, -130, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
        
        ax_inset.set_xlabel(" ")
        # Draw an arrow pointing towards the inset plot
        ax.annotate("", xy=(inset_center_x, inset_center_y), xytext=(box_center_lon, box_center_lat),
                        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2), fontsize=12, transform=ccrs.PlateCarree())
        
        fig.savefig(f"world_map_MYDS+climate_1950-2014_{model}vs_MMM(std-models+members)_diffSTD_v3.pdf", bbox_inches="tight")
        fig.savefig(f"world_map_MYDS+climate_1950-2014_{model}vs_MMM(std-models+members)_diffSTD_v3.jpg", bbox_inches="tight", dpi=1200)

#%% Make a plot for the internal variability, by comparing one member to the MMM

std = n_droughts_all_models.std(dim=("member", "model"))
members = np.arange(0,10)

n_dm = n_droughts_all_models.where(~n_droughts_all_models.isnull().all(dim=("model", "member")))
mean_all = n_dm.mean(dim=("model", "member"))        
diffs = abs(n_dm - mean_all)
diffs_zeroed = diffs.fillna(0)
internal_var = diffs_zeroed.mean(dim=("model", "member"))/std
internal_var = internal_var.where(~n_droughts_all_models.isnull().all(dim=("model","member")))

diff_ERA = (abs(n_droughts_all_models.mean(dim=("member", "model"))-n_droughts_ERA))/std
bias_ERA = diff_ERA-internal_var

#Plot
fig, axes = plt.subplots(3,1, subplot_kw=dict(projection=ccrs.EqualEarth(central_longitude=11)), figsize=(12,12))
plot_kwargs = dict(transform=ccrs.PlateCarree(), cmap="YlOrBr", vmin=0.2, vmax=1.3, extend="both", add_colorbar=False)
internal_var.where(~lc_ERA.isin([220, 210, 200, 150])).plot(ax=axes[0], **plot_kwargs)
axes[0].set_title("a) Internal variability")
diff_ERA.plot(ax=axes[1], **plot_kwargs)
axes[1].set_title("b) Diff. ERA5 versus MMM")
a = bias_ERA.plot(ax=axes[2], **plot_kwargs)
axes[2].set_title("c) Difference - internal variability = bias ERA5 versus MMM")
#Add colourbar
cbar = fig.colorbar(a, ax=axes, orientation="horizontal",fraction=0.013, pad=0.05, aspect=40, extend="both")
cbar.set_label(r"Positive and relative difference in MYDs ($\sigma_{MMM}$)")

for ax in axes.flatten():
    ax.coastlines()
    ax.set_extent([-182, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
    ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="grey", alpha=0.5)
    ax.axis("off")


   
        
fig.savefig("Bias+internal_var_ERA5_vs_MMM_v2.pdf", bbox_inches="tight")
fig.savefig("Bias+internal_var_ERA5_vs_MMM_v2.jpg", bbox_inches="tight", dpi=1200)
