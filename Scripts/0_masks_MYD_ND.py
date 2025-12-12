#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:13:29 2024

@author: Jonna van Mourik
Calculate MYD and ND periods
"""
import xarray as xr
import sys
sys.path.append("/your/dir/") 
from functions import MYD, ND, mask_MYD, mask_ND
import numpy as np

#Data
dir = "/your/dir/"
model = "MPI-ESM1-2-LR" 

SPEI_AUS = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_AUS.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
SPEI_WEU = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_WEU.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
SPEI_CAL = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_CAL.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
SPEI_IND = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_IND.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
SPEI_SA = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_SA.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
SPEI_ARG = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/1950-2014/SPEI12_monthly_1950_2014_r*_"+str(model)+"_ARG.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))

#Masks
mask_AUS = xr.open_dataset(dir+"masks/"+str(model)+"/mask_AUS_"+str(model)+".nc").Band1
mask_WEU = xr.open_dataset(dir+"masks/"+str(model)+"/mask_WEU_"+str(model)+".nc").Band1
mask_IND = xr.open_dataset(dir+"masks/"+str(model)+"/mask_IND_"+str(model)+".nc").Band1
mask_SA = xr.open_dataset(dir+"masks/"+str(model)+"/mask_SA_"+str(model)+".nc").Band1
mask_CAL = xr.open_dataset(dir+"masks/"+str(model)+"/mask_CAL_"+str(model)+".nc").Band1
mask_ARG = xr.open_dataset(dir+"masks/"+str(model)+"/mask_ARG_"+str(model)+".nc").Band1

#Australia
mask_MYD_AUS = mask_MYD(SPEI_AUS, "AUS", oneyear=True)
mask_ND_AUS = mask_ND(SPEI_AUS, "AUS", oneyear=True)

#South Africa
mask_MYD_SA = mask_MYD(SPEI_SA, "SA", oneyear=True)
mask_ND_SA = mask_ND(SPEI_SA, "SA", oneyear=True)

#California
mask_MYD_CAL = mask_MYD(SPEI_CAL, "CAL", oneyear=True)
mask_ND_CAL = mask_ND(SPEI_CAL, "CAL", oneyear=True)
      
#Western Europe
mask_MYD_WEU = mask_MYD(SPEI_WEU, "WEU", oneyear=True)
mask_ND_WEU = mask_ND(SPEI_WEU, "WEU", oneyear=True)

#Middle Argentina
mask_MYD_ARG = mask_MYD(SPEI_ARG, "ARG", oneyear=True)
mask_ND_ARG = mask_ND(SPEI_ARG, "ARG", oneyear=True)

#India
mask_MYD_IND = mask_MYD(SPEI_IND, "IND", oneyear=True)
mask_ND_IND = mask_ND(SPEI_IND, "IND", oneyear=True)

#%% Save masks
def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
    
reg_name = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]    
reg_mask_MYD = [mask_MYD_CAL, mask_MYD_WEU, mask_MYD_IND, mask_MYD_ARG, mask_MYD_SA, mask_MYD_AUS]
reg_mask_ND = [mask_ND_CAL, mask_ND_WEU, mask_ND_IND, mask_ND_ARG, mask_ND_SA, mask_ND_AUS]
for i in range(len(reg_name)):        
    reg_mask_MYD[i].to_netcdf(dir+"masks/"+str(model)+"/mask_MYD_"+str(model)+"_"+str(reg_name[i])+"_1850-2014_1yprior.nc")
    reg_mask_ND[i].to_netcdf(dir+"masks/"+str(model)+"/mask_ND_"+str(model)+"_"+str(reg_name[i])+"_1850-2014_1yprior.nc")
