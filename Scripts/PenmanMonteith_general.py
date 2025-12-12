#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:01:38 2024
@author: Jonna van Mourik

PET for any dataset
"""
import numpy as np
import pyet as pyet
import xarray as xr

#%% Specify years, members, model
model = "CESM2"
#members = np.arange(1,3,1)
members = [1]
startyear = ["1850", "1901", "1951", "1981"]
endyear = ["1900", "1950", "1980", "2014"]
grid = "gn"
#%% Calculate PET
for member in members:
    for i in range(len(startyear)): 
      print(member)
      print(startyear[i])        
      #Load in datasets    
      dir = "/your/dir/CMIP6/"+str(model)+"/"
      if model != "CESM2":
          tmax = xr.open_mfdataset(dir+"tasmax_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").tasmax.sel(time=slice(startyear[i], endyear[i]))-273.15 # Daily maximum temperature [°C]    
          tmin = xr.open_mfdataset(dir+"tasmin_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").tasmin.sel(time=slice(startyear[i], endyear[i]))-273.15 # Daily minimum temperature [°C]
      tmean = xr.open_mfdataset(dir+"tas_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").tas.sel(time=slice(startyear[i], endyear[i]))-273.15 # Daily mean temperature [°C]
      print("Temperature loaded in")
      
      rh_mean = xr.open_mfdataset(dir+"hurs_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").hurs.sel(time=slice(startyear[i], endyear[i])) # Daily mean relative humidity [%]
      if (member==1 or member==2) and model=="ACCESS-ESM1-5":
          rh_max = xr.open_dataset(dir+"hursmax_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1950-2014.nc").hursmax.sel(time=slice(startyear[i], endyear[i])) # Daily max relative humidity [%]
          rh_min = xr.open_mfdataset(dir+"hursmin_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").hursmin.sel(time=slice(startyear[i], endyear[i])) # Daily min relative humidity [%]
      elif model != "CESM2":
          rh_max = xr.open_mfdataset(dir+"hursmax_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").hursmax.sel(time=slice(startyear[i], endyear[i])) # Daily max relative humidity [%]
          rh_min = xr.open_mfdataset(dir+"hursmin_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").hursmin.sel(time=slice(startyear[i], endyear[i])) # Daily min relative humidity [%]

      print("Relative humidity calculated")
      
      if model == "MIROC6":
          u10 = xr.open_dataset(dir+"uas_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1850-2014.nc").uas.sel(time=slice(startyear[i], endyear[i]))
          v10 = xr.open_dataset(dir+"vas_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1850-2014.nc").vas.sel(time=slice(startyear[i], endyear[i]))
          uz = np.sqrt(u10**2+v10**2)  # Wind speed at 10 m [m/s]
      else:
          uz = xr.open_mfdataset(dir+"sfcWind_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").sfcWind.sel(time=slice(startyear[i], endyear[i]))  # Wind speed at 10 m [m/s]
      z = 10  # Height of wind measurement [m]
      wind_fao56 = uz * 4.87 / np.log(67.8*z-5.42)  # wind speed at 2 m after Allen et al., 1998
      print("Wind loaded in and calculated for 2m instead of 10m")
      
      if model == "CESM2":
          p = xr.open_mfdataset(dir+"psl_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").psl.sel(time=slice(startyear[i], endyear[i]))*1e-3 #Surface pressure [kPa]
      else:
          p = xr.open_mfdataset(dir+"psl_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").psl.sel(time=slice(startyear[i], endyear[i]))*1e-3 #Surface pressure [kPa]

      print("Surface pressure loaded in")
      if model == "ACCESS-ESM1-5":
          drs = xr.open_dataset(dir+"rsds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1950-2014.nc").rsds.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 # Compute solar radiation [MJ/m2day]
          drt = xr.open_dataset(dir+"rlds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1950-2014.nc").rlds.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 #thermal radiation [MJ/m2day]
  
          urs = xr.open_dataset(dir+"rsus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1950-2014.nc").rsus.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 # Compute solar radiation [MJ/m2day]
          urt = xr.open_dataset(dir+"rlus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_1950-2014.nc").rlus.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 #thermal radiation [MJ/m2day]
      else:
          drs = xr.open_mfdataset(dir+"rsds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").rsds.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 # Compute solar radiation [MJ/m2day]
          drt = xr.open_mfdataset(dir+"rlds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").rlds.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 #thermal radiation [MJ/m2day]
  
          urs = xr.open_mfdataset(dir+"rsus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").rsus.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 # Compute solar radiation [MJ/m2day]
          urt = xr.open_mfdataset(dir+"rlus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_"+str(grid)+"_*.nc").rlus.sel(time=slice(startyear[i], endyear[i]))*3600*1e-6 #thermal radiation [MJ/m2day]
  
      nrs = drs - urs
      nrt = drt - urt
      rn = nrs + nrt #Turn into + if rt<0! Net radiation
      time = tmean.time
      lat = tmean.lat
      elevation = 2
      print("Radiation loaded in and calculated")
      
      if model == "CESM2":
          pm_fao56 = pyet.pm_fao56(tmean, wind=wind_fao56, rs=nrs, rn=rn, pressure=p, elevation=elevation, lat=lat, rh=rh_mean) #
      else:
          pm_fao56 = pyet.pm_fao56(tmean, wind=wind_fao56, rs=nrs, rn=rn, pressure=p, elevation=elevation, lat=lat, rh=rh_mean, tmax=tmax, tmin=tmin, rhmax=rh_max, rhmin=rh_min) #
      print("Penman-Monteith calculated")
  
      pm_fao56.to_netcdf(dir +"pm_fao56_r"+str(member)+"_"+str(startyear[i])+"-"+str(endyear[i])+"_daily_"+str(model)+"_hist.nc")
      print("Penman-Monteith saved to netCDF")
