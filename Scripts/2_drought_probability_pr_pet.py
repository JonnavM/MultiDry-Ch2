#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:24:53 2025

@author: Jonna van Mourik
Make plot with the probability of droughts based on the PET or PR value for both MMM and ERA5
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import sys
sys.path.append("your/directory/")
from functions import MYD, ND, mask_MYD, mask_ND
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
import pandas as pd
import cftime
#%% Import ERA5 data
dir = "your/dir/"
#Data
SPEI_AUS = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").sel(time=slice("1951", "2023"))
SPEI_WEU = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").sel(time=slice("1951", "2023"))
SPEI_CAL = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").sel(time=slice("1951", "2023"))
SPEI_IND = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_IND.nc").sel(time=slice("1951", "2023"))
SPEI_SA = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_SA.nc").sel(time=slice("1951", "2023"))
SPEI_ARG = xr.open_dataarray(dir+"SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").sel(time=slice("1951", "2023"))

# Masks
mask_AUS = xr.open_dataarray(dir+"mask_AUS.nc")
mask_WEU = xr.open_dataarray(dir+"mask_WEU.nc")
mask_IND = xr.open_dataarray(dir+"mask_IND.nc")
mask_SA = xr.open_dataarray(dir+"mask_SA.nc") 
mask_CAL = xr.open_dataarray(dir+"mask_CAL.nc")
mask_ARG = xr.open_dataarray(dir+"mask_ARG.nc")

#Australia
mask_MYD_AUS = mask_MYD(SPEI_AUS, "AUS")
mask_ND_AUS = mask_ND(SPEI_AUS, "AUS")

#South Africa
mask_MYD_SA = mask_MYD(SPEI_SA, "SA")
mask_ND_SA = mask_ND(SPEI_SA, "SA")

#California
mask_MYD_CAL = mask_MYD(SPEI_CAL, "CAL")
mask_ND_CAL = mask_ND(SPEI_CAL, "CAL")
      
#Western Europe
mask_MYD_WEU = mask_MYD(SPEI_WEU, "WEU")
mask_ND_WEU = mask_ND(SPEI_WEU, "WEU")

#Middle Argentina
mask_MYD_ARG = mask_MYD(SPEI_ARG, "ARG")
mask_ND_ARG = mask_ND(SPEI_ARG, "ARG")

#India
mask_MYD_IND = mask_MYD(SPEI_IND, "IND")
mask_ND_IND = mask_ND(SPEI_IND, "IND")

#Lats and lons per region
lat_WEU = slice(45, 55) 
lon_WEU = slice(-1, 13)
lat_IND = slice(22, 32)
lon_IND = slice(72, 90)
lat_AUS = slice(-40, -20)
lon_AUS = slice(135, 155)
lat_SA = slice(-33, -21)
lon_SA = slice(15, 31)
lat_ARG = slice(-45, -25)
lon_ARG = slice(-75, -55)
lat_CAL = slice(32, 41.5)
lon_CAL = slice(-124, -115)

#%% Same for MMM
models = ["MIROC6", "MPI-ESM1-2-LR", "CanESM5", "ACCESS-ESM1-5", "CESM2", "EC-Earth3"]
title_reg = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]
afk = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]

startyear = "1950"
SPEI_reg_model = []
mask_MYD_reg_model = []
mask_ND_reg_model = []
for model in models:
    SPEI_AUS_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_AUS.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_WEU_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_WEU.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_CAL_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_CAL.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_IND_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_IND.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_SA_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_SA.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    SPEI_ARG_MMM = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/"+str(startyear)+"-2014/SPEI12_monthly_"+str(startyear)+"_2014_r*_"+str(model)+"_ARG.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__.sel(time=slice(startyear, "2014"))
    
    #Australia
    mask_MYD_AUS_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_AUS_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_AUS_MMM = xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_AUS_1850-2014.nc").__xarray_dataarray_variable__
    
    #South Africa
    mask_MYD_SA_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_SA_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_SA_MMM = xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_SA_1850-2014.nc").__xarray_dataarray_variable__
    
    #California
    mask_MYD_CAL_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_CAL_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_CAL_MMM = xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_CAL_1850-2014.nc").__xarray_dataarray_variable__
          
    #Western Europe
    mask_MYD_WEU_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_WEU_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_WEU_MMM = xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_WEU_1850-2014.nc").__xarray_dataarray_variable__
    
    #Middle Argentina
    mask_MYD_ARG_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_ARG_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_ARG_MMM = xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_ARG_1850-2014.nc").__xarray_dataarray_variable__
    
    #India
    mask_MYD_IND_MMM = xr.open_dataset(dir+f"masks/{model}/mask_MYD_{model}_IND_1850-2014.nc").__xarray_dataarray_variable__
    mask_ND_IND_MMM =xr.open_dataset(dir+f"masks/{model}/mask_ND_{model}_IND_1850-2014.nc").__xarray_dataarray_variable__
    
    #Combine in lists
    SPEI_reg_model.append([SPEI_CAL_MMM, SPEI_WEU_MMM, SPEI_IND_MMM, SPEI_ARG_MMM, SPEI_SA_MMM, SPEI_AUS_MMM])
    mask_MYD_reg_model.append([mask_MYD_CAL_MMM, mask_MYD_WEU_MMM, mask_MYD_IND_MMM, mask_MYD_ARG_MMM, mask_MYD_SA_MMM, mask_MYD_AUS_MMM])
    mask_ND_reg_model.append([mask_ND_CAL_MMM, mask_ND_WEU_MMM, mask_ND_IND_MMM, mask_ND_ARG_MMM, mask_ND_SA_MMM, mask_ND_AUS_MMM])
    
SPEI_reg_MMM = xr.DataArray(SPEI_reg_model, dims=("model", "region", "member", "time"), coords={"region":afk, "model":models, "member":np.arange(0,10), "time":SPEI_AUS_MMM.time}).compute()
mask_MYD_reg_MMM = xr.DataArray(mask_MYD_reg_model, dims=("model", "region", "member", "time"), coords={"region":afk, "model":models, "member":np.arange(0,10), "time":mask_MYD_AUS_MMM.time}).compute()
mask_ND_reg_MMM = xr.DataArray(mask_ND_reg_model, dims=("model", "region", "member", "time"), coords={"region":afk, "model":models, "member":np.arange(0,10), "time":mask_ND_AUS_MMM.time}).compute()

#%% Download ERA pet and pr
pr = xr.open_dataarray(dir+"ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc")*1000#.sel(time=slice("1950", "2022"), expver=1)*1000
pet = xr.open_dataset(dir+"PET/PenmanMonteith/ERA5/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56
pet["time"]=pr["time"]

#Rolling mean
def rolling12(data):
    return data.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))

#Anomaly
def std_anom(data):
    climatology_mean = data.groupby('time.month').mean('time')
    climatology_std = data.groupby('time.month').std('time')
    climatology_std = climatology_std.where(climatology_std != 0, float('nan'))
    stand_anomalies = xr.apply_ufunc(lambda x, m, s: (x - m) / s, data.groupby('time.month'), climatology_mean, climatology_std,dask = 'allowed', vectorize = True)
    print ('anomalies done')
    return stand_anomalies

def anom(data):
    climatology_mean = data.groupby('time.month').mean('time')
    stand_anomalies = xr.apply_ufunc(lambda x, m: (x - m), data.groupby('time.month'), climatology_mean, dask = 'allowed', vectorize = True)
    print ('anomalies done')
    return stand_anomalies

SPEI_reg = [SPEI_CAL, SPEI_WEU, SPEI_IND, SPEI_ARG, SPEI_SA, SPEI_AUS]
mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS] 
mask_MYD_reg = [mask_MYD_CAL, mask_MYD_WEU, mask_MYD_IND, mask_MYD_ARG, mask_MYD_SA, mask_MYD_AUS]

#Change units to mm/month
days_in_month = pr.time.dt.days_in_month
pr_month = pr*days_in_month
pet_month = pet*days_in_month
pr_month.attrs['units'] = 'mm/month'
pet_month.attrs['units'] = 'mm/month'
pr_month_norm = std_anom(pr_month)
pet_month_norm = std_anom(pet_month)
pr_month_anom = anom(pr_month)
pet_month_anom = anom(pet_month)

#%% Download CMIP6 pet and pr

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
        pr_MMM = xr.open_mfdataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r*i1p1f1_"+str(grid)+"_1850-2014_1x1.nc", preprocess=preprocess, combine='nested', concat_dim='member').pr.sel(time=slice("1950", "2014"))*3600*24
        pet_MMM = xr.open_mfdataset(dir+"PET/"+str(model)+"/1x1/pm_fao56_r*_1850-2014_monthly_"+str(model)+"_hist_1x1.nc", preprocess=preprocess, combine="nested", concat_dim="member").PM_FAO_56.sel(time=slice("1950", "2014"))
    else:
        print(model)
        grid = "gn"
        pr_MMM = xr.open_mfdataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r*i1p1f1_"+str(grid)+"_1850-2014_1x1.nc", combine='nested', concat_dim='member').pr.sel(time=slice("1950", "2014"))*3600*24
        pet_MMM = xr.open_mfdataset(dir+"PET/"+str(model)+"/1x1/pm_fao56_r*_1850-2014_monthly_"+str(model)+"_hist_1x1.nc", combine='nested', concat_dim='member').PM_FAO_56.sel(time=slice("1950", "2014"))
    #Change units to mm/month
    days_in_month = pr_MMM.time.dt.days_in_month
    pr_month_MMM = (pr_MMM*days_in_month).resample(time="1MS").mean()
    days_in_month = pet_MMM.time.dt.days_in_month
    pet_month_MMM = (pet_MMM*days_in_month).resample(time="1MS").mean()
    #pet_month["lat"]=pr_month["lat"]
    pr_month_MMM.attrs['units'] = 'mm/month'
    pet_month_MMM.attrs['units'] = 'mm/month'
    pr_list.append(convert_time(pr_month_MMM))
    pet_list.append(convert_time(pet_month_MMM))
    print(pr_month_MMM.time)
    
pr_month_all = xr.concat(pr_list, dim="model", coords="minimal")
pet_month_all = xr.concat(pet_list, dim="model", coords="minimal")

pr_month_all = pr_month_all.assign_coords(model=("model", models), member=("member", np.arange(0,10)))#.compute()
pet_month_all = pet_month_all.assign_coords(model=("model", models), member=("member", np.arange(0,10)))#.compute()
#normalize
pr_month_all_norm = std_anom(pr_month_all).compute()
pet_month_all_norm = std_anom(pet_month_all).compute()

#Load in masks for CMIP
mask_AUS_MMM = xr.open_dataset(dir+"masks/1x1/mask_AUS_1x1.nc").Band1
mask_WEU_MMM = xr.open_dataset(dir+"masks/1x1/mask_WEU_1x1.nc").Band1
mask_IND_MMM = xr.open_dataset(dir+"masks/1x1/mask_IND_1x1.nc").Band1
mask_SA_MMM = xr.open_dataset(dir+"masks/1x1/mask_SA_1x1.nc").Band1
mask_CAL_MMM = xr.open_dataset(dir+"masks/1x1/mask_CAL_1x1.nc").Band1
mask_ARG_MMM = xr.open_dataset(dir+"masks/1x1/mask_ARG_1x1.nc").Band1

mask_reg_MMM = [mask_CAL_MMM, mask_WEU_MMM, mask_IND_MMM, mask_ARG_MMM, mask_SA_MMM, mask_AUS_MMM]

#%% Make definition of everything that needs to be calculated
def pdf_droughts(index_month, i, model=None):
    """
    Input: index_month, i, model
    if model is entered as an input, only one model is evaluated, otherwise the MMM is taken.
    Output: x, index_pdf_normal_values, index_pdf_drought_values, index_pdf_MYD_values
    """
    ###Take rolling mean of index
    if "model" not in index_month.dims:
        print("Reanalysis data")
        index_12 = rolling12(index_month.where(mask_reg[i]==1).mean(dim=("lat", "lon")))
        # Select non-droughts, droughts, and MYDs
        index_normal = index_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]>-1))
        index_drought = index_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
        index_MYD = index_12.where(mask_MYD_reg[i]==True)
    else:
        # Select non-droughts, droughts, and MYDs
        if model!=None:
            print("Evaluate one model")
            index_12 = rolling12(index_month.sel(model=model).where(mask_reg_MMM[i]==1).mean(dim=("lat", "lon")))
            index_normal = index_12.where((mask_MYD_reg_MMM.sel(region=afk[i], model=model)==False) & (SPEI_reg_MMM.sel(region=afk[i], model=model)>-1)).stack(z=("member", "time"))
            index_drought = index_12.where((mask_MYD_reg_MMM.sel(region=afk[i], model=model)==False) & (SPEI_reg_MMM.sel(region=afk[i], model=model)<=-1)).stack(z=("member", "time"))
            index_MYD = index_12.where(mask_MYD_reg_MMM.sel(region=afk[i], model=model)==True).stack(z=("member", "time"))
        if model==None:
            print("Evaluate MMM")
            index_12 = rolling12(index_month.where(mask_reg_MMM[i]==1).mean(dim=("lat", "lon")))
            index_normal = index_12.where((mask_MYD_reg_MMM.sel(region=afk[i])==False) & (SPEI_reg_MMM.sel(region=afk[i])>-1)).stack(z=("model", "member", "time"))
            index_drought = index_12.where((mask_MYD_reg_MMM.sel(region=afk[i])==False) & (SPEI_reg_MMM.sel(region=afk[i])<=-1)).stack(z=("model", "member", "time"))
            index_MYD = index_12.where(mask_MYD_reg_MMM.sel(region=afk[i])==True).stack(z=("model", "member", "time"))

    ###Part where the probabilities are plotted
    # Remove NaN values
    index_normal = index_normal[~np.isnan(index_normal)]
    index_drought = index_drought[~np.isnan(index_drought)]
    index_MYD = index_MYD[~np.isnan(index_MYD)]
    
    ###Calculate PDFs
    index_pdf_normal = gaussian_kde(index_normal)
    index_pdf_drought = gaussian_kde(index_drought)
    index_pdf_MYD = gaussian_kde(index_MYD)
    
    ### Define a range of PET values for the x-axis
    x_index = np.linspace(min(index_normal.min(), index_drought.min(), index_MYD.min()), 
                    max(index_normal.max(), index_drought.max(), index_MYD.max()), 1000)
    
    ### Evaluate PDFs
    index_pdf_normal_values = index_pdf_normal(x_index)
    index_pdf_drought_values = index_pdf_drought(x_index)
    index_pdf_MYD_values = index_pdf_MYD(x_index)
    
    ### Normalize the PDFs so they stack to 1
    index_total_pdf_values = index_pdf_normal_values + index_pdf_drought_values + index_pdf_MYD_values
    index_pdf_normal_values /= index_total_pdf_values
    index_pdf_drought_values /= index_total_pdf_values
    index_pdf_MYD_values /= index_total_pdf_values
    
    return x_index, index_pdf_normal_values, index_pdf_drought_values, index_pdf_MYD_values

    
#%% Make a plot for the probability where you can see the different models lined up within one plot
fontsize = 12
for i in range(6):
    fig, ax = plt.subplots(2,2, figsize=(6,2), sharex=True, sharey=True, gridspec_kw={"hspace":0.2, "wspace":0.1})
    #First calculate and plot everything for ERA5
    x_pr, pr_pdf_normal_values, pr_pdf_drought_values, pr_pdf_MYD_values = pdf_droughts(pr_month_norm, i)
    x_pet, pet_pdf_normal_values, pet_pdf_drought_values, pet_pdf_MYD_values = pdf_droughts(pet_month_norm, i)
    
    ### Plot the stacked probability plot
    ax[0,0].fill_between(x_pr, 0, pr_pdf_normal_values, label='Other', color='tab:blue')
    ax[0,0].fill_between(x_pr, pr_pdf_normal_values, pr_pdf_normal_values + pr_pdf_drought_values, label='ND', color='orange')
    ax[0,0].fill_between(x_pr, pr_pdf_normal_values + pr_pdf_drought_values, pr_pdf_normal_values + pr_pdf_drought_values + pr_pdf_MYD_values, label='MYD', color='red')#, alpha=0.5)
    
    ax[0,1].fill_between(x_pet, 0, pet_pdf_normal_values, label='Other', color='tab:blue')#, alpha=0.5)
    ax[0,1].fill_between(x_pet, pet_pdf_normal_values, pet_pdf_normal_values + pet_pdf_drought_values, label='ND', color='orange')#, alpha=0.5)
    ax[0,1].fill_between(x_pet, pet_pdf_normal_values + pet_pdf_drought_values, pet_pdf_normal_values + pet_pdf_drought_values + pet_pdf_MYD_values, label='MYD', color='red')#, alpha=0.5)
    #ax[0,1].legend(loc=[0.5, 1.2])
    ###Draw line at P(Not dry)=0 and at P(MYD)=1
    if (pr_pdf_normal_values<=0.005).any()==True:
        ax[0,0].axvline(max(x_pr[pr_pdf_normal_values<=0.005]), color="black", linestyle="dashed")
    if (pr_pdf_MYD_values>=0.995).any()==True:
        ax[0,0].axvline(max(x_pr[pr_pdf_MYD_values>=0.995]), color="black")
        
    if (pet_pdf_normal_values<=0.005).any()==True:
        ax[0,1].axvline(min(x_pet[pet_pdf_normal_values<=0.005]), color="black", linestyle="dashed")
    if (pet_pdf_MYD_values>=0.995).any()==True:
        ax[0,1].axvline(min(x_pet[pet_pdf_MYD_values>=0.995]), color="black")
    
    x_pr_MMM, pr_pdf_normal_values_MMM, pr_pdf_drought_values_MMM, pr_pdf_MYD_values_MMM = pdf_droughts(index_month=pr_month_all_norm, i=i)
    x_pet_MMM, pet_pdf_normal_values_MMM, pet_pdf_drought_values_MMM, pet_pdf_MYD_values_MMM = pdf_droughts(index_month=pet_month_all_norm, i=i)
    
    #Now loop through all models for the other panels
    for model in models:
        #MMM
        x_pr_SM, pr_pdf_normal_values_SM, pr_pdf_drought_values_SM, pr_pdf_MYD_values_SM = pdf_droughts(pr_month_all_norm, i, model=model)
        x_pet_SM, pet_pdf_normal_values_SM, pet_pdf_drought_values_SM, pet_pdf_MYD_values_SM = pdf_droughts(pet_month_all_norm, i, model=model)

        #Also for MMM in the main axes
        ax[1,0].fill_between(x_pr_SM, 0, pr_pdf_normal_values_SM, label='Not dry', color='tab:blue', alpha=0.25)
        ax[1,0].fill_between(x_pr_SM, pr_pdf_normal_values_SM, pr_pdf_normal_values_SM + pr_pdf_drought_values_SM, label='ND', color='orange', alpha=0.25)
        ax[1,0].fill_between(x_pr_SM, pr_pdf_normal_values_SM + pr_pdf_drought_values_SM, pr_pdf_normal_values_SM + pr_pdf_drought_values_SM + pr_pdf_MYD_values_SM, label='MYD', color='red', alpha=0.25)#, alpha=0.5)
        
        ax[1,1].fill_between(x_pet_SM, 0, pet_pdf_normal_values_SM, label='Not dry', color='tab:blue', alpha=0.25)#, alpha=0.5)
        ax[1,1].fill_between(x_pet_SM, pet_pdf_normal_values_SM, pet_pdf_normal_values_SM + pet_pdf_drought_values_SM, label='ND', color='orange', alpha=0.2)#, alpha=0.5)
        ax[1,1].fill_between(x_pet_SM, pet_pdf_normal_values_SM + pet_pdf_drought_values_SM, pet_pdf_normal_values_SM + pet_pdf_drought_values_SM + pet_pdf_MYD_values_SM, label='MYD', color='red', alpha=0.2)#, alpha=0.5)
        
    #also for MMM
    if (pr_pdf_normal_values_MMM<=0.005).any()==True:
        ax[1,0].axvline(max(x_pr_MMM[pr_pdf_normal_values_MMM<=0.005]), color="black", linestyle="dashed")
    if (pr_pdf_MYD_values_MMM>=0.995).any()==True:
        ax[1,0].axvline(max(x_pr_MMM[pr_pdf_MYD_values_MMM>=0.995]), color="black")
        
    if (pet_pdf_normal_values_MMM<=0.005).any()==True:
        ax[1,1].axvline(min(x_pet_MMM[pet_pdf_normal_values_MMM<=0.005]), color="black", linestyle="dashed")
    if (pet_pdf_MYD_values_MMM>=0.995).any()==True:
        ax[1,1].axvline(min(x_pet_MMM[pet_pdf_MYD_values_MMM>=0.995]), color="black")
        
    #Format of plot
    fig.suptitle(afk[i], fontsize=fontsize)
    fig.subplots_adjust(left=0.15)
    fig.supylabel("Probability", fontsize=fontsize)
    ax[1,0].set_ylabel("MMM ", fontsize=fontsize)
    ax[0,0].set_ylabel("ERA5", fontsize=fontsize)
    
    # Add the legend to the plot
    ax[1,0].set_xlabel(r"St. anom. PR", fontsize=fontsize)
    ax[1,1].set_xlabel(r"St. anom. PET", fontsize=fontsize)
    
    for axes in ax.flat:
        axes.set_ylim(0, 1)
        axes.set_xlim(-2.1, 2.1)
    fig.savefig(f"prob_{afk[i]}_AllModels_normalised_v5.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"prob_{afk[i]}_AllModels_normalised_v5.pdf", bbox_inches="tight")
    plt.show() 
 
