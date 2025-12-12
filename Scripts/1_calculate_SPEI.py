#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:37:12 2024

@author: Jonna van Mourik
"""

import xclim
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xclim import indices
from xclim.core import units
from xclim.indices import standardized_precipitation_evapotranspiration_index
import spei as si  # si for standardized index
import pandas as pd

xr.set_options(keep_attrs=True)

#%% Define SPEI
model = "ACCESS-ESM1-5"
dir = "/your/dir/"

def SPEI_region(region_name, prec, pet, spei_period, offset, cal_start, cal_end, dir2):
    if region_name == "IND": #IND, AUS, WEU, SA, ARG, CAL
        reg_lat = slice(10,47)
        reg_lon = slice(60,110)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_IND_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "AUS":
        reg_lat = slice(-50, -10)
        reg_lon = slice(120, 165)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_AUS_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "WEU":
        reg_lat = slice(20, 70)
        reg_lon = slice(-30,30)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_WEU_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "SA":
        reg_lat = slice(-45,-12)
        reg_lon = slice(2, 50)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_SA_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "ARG":
        reg_lat = slice(-45, -25)
        reg_lon = slice(-80, -55)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_ARG_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "CAL":
        reg_lat = slice(15, 60)
        reg_lon = slice(-124, -115)
        region_mask = xr.open_dataarray(dir+"masks/1x1/mask_CAL_1x1.nc").interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)        
        
    prec_region = prec.where(region_mask>=0.9)
    pet_region = pet.where(region_mask>0.9)
    
    prec_region_mean = prec_region.mean(dim = ["lon","lat"])
    pet_region_mean = pet_region.mean(dim = ["lon","lat"])
    
    pe_region_mean = prec_region_mean.assign_attrs(units='mm/d') - pet_region_mean.assign_attrs(units='mm/d')
    # Convert cftime to datetime64[ns]
    pe_region_mean = pe_region_mean.convert_calendar("standard", use_cftime=False)

    print("calculating spei")
    SPEI = standardized_precipitation_evapotranspiration_index(pe_region_mean, window = spei_period, dist = "fisk",freq= "MS", offset=offset,  cal_start = cal_start, cal_end = cal_end)

    del SPEI.attrs['freq']
    del SPEI.attrs['time_indexer']
    del SPEI.attrs['units']
    del SPEI.attrs['offset']
    print("saving")
    SPEI.to_netcdf(path = dir+"SPEI/"+str(model)+"/" + dir2)
    print("done")
    
def SPEI_global(prec, pet, spei_period, offset, cal_start, cal_end, dir3):
    pe = prec.assign_attrs(units='mm/d') - pet.assign_attrs(units='mm/d')
    print("Calculating SPEI")
    SPEI = standardized_precipitation_evapotranspiration_index(pe, window = spei_period, dist = "fisk",freq= "MS", offset=offset,  cal_start = cal_start, cal_end = cal_end)
    del SPEI.attrs['freq']
    del SPEI.attrs['time_indexer']
    del SPEI.attrs['units']
    del SPEI.attrs['offset']
    print("saving")
    SPEI.to_netcdf(path = dir+"SPEI/"+str(model)+"/" + dir3)
    print("done")
    
#%% Load in data
# landmask, since we don't need data over the oceans 
landmask = xr.open_dataset(dir+"sftlf_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn_1x1.nc").sftlf 

#%% Calculate per member
members = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
years = ["1950"]

for member in members:
    for year in years:
        print(member, year)
        # Precipitation, is in mm/s, but we need it in mm/day.
        total_prec_mm = (xr.open_mfdataset(dir+"CMIP6/"+str(model)+"/1x1/pr_Amon_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_1850-2014_1x1.nc").pr*3600*24).resample(time="MS").mean()
        landmask_interpolated = landmask.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest')
        total_prec_mm = total_prec_mm.where(landmask_interpolated>=50)
        # PET, resample to monthly values
        pet = xr.open_mfdataset(dir+"PET/"+str(model)+"/1x1/pm_fao56_r"+str(member)+"_*_daily_"+str(model)+"_hist_1x1.nc").PM_FAO_56.where(landmask_interpolated>=50).resample(time="1MS").mean()
        # Calculate SPEI 
        # Set the offset, specify start- and end of the calibration period. Mean is zero between these dates
        prec = total_prec_mm
        pet = pet
        spei_period = 12
        offset = '20 mm/d'
        cal_start = str(year)+"-01-01"
        cal_end = "2014-12-31"
        
        #For regional values
        region_name = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
        dir2 = ["SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_CAL_v2.nc", "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_WEU_v2.nc",
               "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_IND_v2.nc", "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_ARG_v2.nc", 
               "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_SA_v2.nc", "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_AUS_v2.nc"]
        
        for i in range(len(region_name)):
            print(region_name[i])
            SPEI_region(region_name[i], prec, pet, spei_period, offset, cal_start, cal_end, dir2[i])
        
        #For global values
        dir3 = "SPEI12_monthly_"+str(year)+"_2014_r"+str(member)+"_"+str(model)+"_1x1_v2.nc"
        
        SPEI_global(prec, pet, spei_period, offset, cal_start, cal_end, dir3)

        
