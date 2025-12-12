#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:05:00 2024

@author: Jonna van Mourik
Drought functions 
"""

import pandas as pd
import xarray as xr
import numpy as np

def MYD(spei, region_name):
    """
    Function to calculate the start, end and length of multi-year droughts. 
    Input: timeseries of SPEI, and a string with the region name. 
    Output: start_date, end_date, length
    """
    print(region_name)
    
    # Initialize lists to store results for all members
    member_list = []

    # Check if SPEI dataset has a 'member' dimension
    if 'member' in spei.dims:
        members = spei.member.values
    else:
        members = [None]  # No member dimension

    # Loop over each member
    for member in members:
        print(f"Processing member: {member}" if member is not None else "Processing single dataset")

        i = 0
        start_date = []
        end_date = []
        length = []

        # Select the specific member, if applicable
        spei_member = spei.sel(member=member) if member is not None else spei

        print("Check for droughts within spei<=-1")
        for t in spei_member.time:
            if spei_member.sel(time=t).mean() <= -1:
                i += 1
                if i == 1:
                    start_drought = t.values
                    start = t
                elif t == spei_member.time[-1] and i >= 12:
                    start_date.append(start)
                    end_date.append(t)
                    length.append(i)
                    print("From ", start_drought, " until ", t.values, ". Duration is ", i, " months.")
            else:
                if i >= 12:
                    start_date.append(start)
                    end = t.values - pd.DateOffset(months=1)
                    end_date.append(spei_member.time.sel(time=end))
                    length.append(i)
                    print("From ", start_drought, " until ", end, ". Duration is ", i, " months.")
                i = 0
                continue
        member_list.append([start_date, end_date, length])
    return member_list

def ND(spei, region_name):
    """
    Function to calculate the start, end and length of shorter (<12 months) droughts. 
    Input: timeseries of SPEI, and a string with the region name. 
    Output: start_date, end_date, length
    """
    print(region_name)
    
    # Initialize lists to store results for all members
    member_list = []
    
    # Check if SPEI dataset has a 'member' dimension
    if 'member' in spei.dims:
        members = spei.member.values
    else:
        members = [None]  # No member dimension

    # Loop over each member
    for member in members:
        print(f"Processing member: {member}" if member is not None else "Processing single dataset")

        i = 0
        start_date = []
        end_date = []
        length = []
        # Select the specific member, if applicable
        spei_member = spei.sel(member=member) if member is not None else spei

        print("Check for droughts within spei<=-1")
        for t in spei_member.time:
            if spei_member.sel(time=t).mean() <= -1:
                i += 1
                if i == 1:
                    start_drought = t.values
                    start = t
                elif t == spei_member.time[-1] and 0 <= i < 12:
                    start_date.append(start)
                    end_date.append(t)
                    length.append(i)
                    print("From ", start_drought, " until ", t.values, ". Duration is ", i, " months.")
            else:
                if 0 < i < 12:
                    start_date.append(start)
                    end = t.values - pd.DateOffset(months=1)
                    end_date.append(spei_member.time.sel(time=end))
                    length.append(i)
                    print("From ", start_drought, " until ", end, ". Duration is ", i, " months.")
                i = 0
                continue
        member_list.append([start_date, end_date, length])
    return member_list

def mask_MYD(SPEI, region_name, oneyear=False, ):
    """
    Function to rephrase MYD to a timeseries with True for the timesteps where a MYD occurs.
    Input: timeseries of SPEI (with member and time dimensions), string with region name.
    Output: timeseries with True for the timesteps where a multi-year drought (MYD) occurs.
    """
    # Get the multi-year drought (MYD) results for all members
    MYD_reg = MYD(SPEI, region_name)
    # Check if SPEI dataset has a 'member' dimension
    if 'member' in SPEI.dims:
        # Initialize the mask DataArray with False values
        mask_MYD_reg = xr.DataArray(False, dims=("member", "time"), coords={"member": SPEI.member, "time": SPEI.time})
        # Loop through each member's drought information
        for member_index, (start_dates, end_dates, lengths) in enumerate(MYD_reg):
            # Loop through each drought period for the current member
            for start, end, length in zip(start_dates, end_dates, lengths):
                # Set True values for the time slices where the drought occurs
                if oneyear==False:
                    mask_MYD_reg[member_index] = mask_MYD_reg[member_index] | (
                        (SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
                elif oneyear==True:
                    mask_MYD_reg[member_index] = mask_MYD_reg[member_index] | (
                        (SPEI.time >= np.datetime64(start.values)-pd.DateOffset(months=12)) & (SPEI.time <= np.datetime64(end.values)))
        return mask_MYD_reg
    else:
        mask_MYD_reg = xr.DataArray(np.full(SPEI.shape, False), dims=("time"), coords={"time": SPEI.time})
        for start, end, length in zip(*MYD_reg[0]):
            # Set True values for the specified time slices
            mask_MYD_reg = mask_MYD_reg | ((SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
        return mask_MYD_reg


def mask_ND(SPEI, region_name, oneyear=False):
    """
    Function to rephrase MYD to a timeseries with True for the timesteps where a normal drought occurs.
    Input: timeseries of SPEI, string with region name
    Output: timeseries with True and False
    """
    ND_reg = ND(SPEI, region_name)
    
    if 'member' in SPEI.dims:
        mask_ND_reg = xr.DataArray(False, dims=("member", "time",),  coords={"member":SPEI.member, "time": SPEI.time})
        for member_index, (start_dates, end_dates, lengths) in enumerate(ND_reg):
            for start, end, length in zip(start_dates, end_dates, lengths):
            # Set True values for the specified time slices
                # Set True values for the time slices where the drought occurs
                if oneyear==False:
                    mask_ND_reg[member_index] = mask_ND_reg[member_index] | (
                        (SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
                elif oneyear==True:
                    mask_ND_reg[member_index] = mask_ND_reg[member_index] | (
                        (SPEI.time >= np.datetime64(start.values)-pd.DateOffset(months=12)) & (SPEI.time <= np.datetime64(end.values)))
        return mask_ND_reg
    else:
        mask_ND_reg = xr.DataArray(np.full(SPEI.shape, False), dims=("time"), coords={"time": SPEI.time})
        for start, end, length in zip(*ND_reg[0]):
            # Set True values for the specified time slices
            mask_ND_reg = mask_ND_reg | ((SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
        return mask_ND_reg

def MYD_gridcell(spei_grid, oneyear=False):
    """
    Identifies multi-year droughts (MYDs) in a gridded SPEI dataset.
    A MYD is defined as a period of at least 12 consecutive months where SPEI <= -1.
    
    Inputs:
    - spei_grid: xarray DataArray with dimensions (member, time, lat, lon) or (time, lat, lon) if no members.
    
    Output:
    - new_mask_data_array: A DataArray where MYD periods are marked as 1, and other times as NaN.
    """
    # Create a mask where SPEI <= -1
    mask = spei_grid <= -1  # Shape: (member, time, lat, lon)
    # Apply a rolling sum over time (rolling window of 12 months)
    rolling_sum = mask.rolling(time=12).sum()
    # Identify where at least 12 consecutive months satisfy SPEI <= -1
    final_mask = rolling_sum >= 12
    # Convert to numpy for element-wise adjustments
    data_array_np = final_mask.values  # Shape: (member, time, lat, lon)

    # Extend the MYD period backward for each identified event
    for member in range(data_array_np.shape[0]):  # Loop over members
        for lat in range(data_array_np.shape[2]):  # Loop over latitudes
            for lon in range(data_array_np.shape[3]):  # Loop over longitudes
                for t in range(data_array_np.shape[1]):  # Loop over time
                    if data_array_np[member, t, lat, lon]:  # If True, extend backwards
                        if oneyear==False:
                            data_array_np[member, max(0, t - 11):t, lat, lon] = True
                        elif oneyear==True:
                            data_array_np[member, max(0, t- 11 - 12):t, lat, lon] = True
                            

    # Convert the modified array back to an xarray DataArray
    modified_data_array = xr.DataArray(data_array_np, coords=spei_grid.coords, dims=spei_grid.dims)

    # Convert boolean mask to NaN (False) and 1 (True)
    new_mask_array = np.where(modified_data_array, 1, np.nan)

    # Convert back to an xarray DataArray
    new_mask_data_array = xr.DataArray(new_mask_array, coords=spei_grid.coords, dims=spei_grid.dims)

    return new_mask_data_array

def ND_gridcell(spei_grid, oneyear=False):
    """
    Identifies normal droughts (NDs) in a gridded SPEI dataset.
    A ND is defined as a period of less than 12 consecutive months where SPEI <= -1.
    
    Inputs:
    - spei_grid: xarray DataArray with dimensions (member, time, lat, lon) or (time, lat, lon) if no members.
    
    Output:
    - new_mask_data_array: A DataArray where MYD periods are marked as 1, and other times as NaN.
    """
    # Create a mask where SPEI <= -1
    mask = spei_grid <= -1  # Shape: (member, time, lat, lon)
    # Apply a rolling sum over time (rolling window of 12 months)
    rolling_sum = mask.rolling(time=12).sum()
    # Identify where less than 12 consecutive months satisfy SPEI <= -1
    final_mask = (rolling_sum >= 1) & (rolling_sum < 12)
    # Convert to numpy for element-wise adjustments
    data_array_np = final_mask.values  # Shape: (member, time, lat, lon)

    # Extend the MYD period backward for each identified event
    for member in range(data_array_np.shape[0]):  # Loop over members
        for lat in range(data_array_np.shape[2]):  # Loop over latitudes
            for lon in range(data_array_np.shape[3]):  # Loop over longitudes
                for t in range(data_array_np.shape[1]):  # Loop over time
                    if data_array_np[member, t, lat, lon]:  # If True, extend backwards
                        if oneyear==False:
                            data_array_np[member, max(0, t - 11):t, lat, lon] = True
                        elif oneyear==True:
                            data_array_np[member, max(0, t- 11 - 12):t, lat, lon] = True
                            

    # Convert the modified array back to an xarray DataArray
    modified_data_array = xr.DataArray(data_array_np, coords=spei_grid.coords, dims=spei_grid.dims)

    # Convert boolean mask to NaN (False) and 1 (True)
    new_mask_array = np.where(modified_data_array, 1, np.nan)

    # Convert back to an xarray DataArray
    new_mask_data_array = xr.DataArray(new_mask_array, coords=spei_grid.coords, dims=spei_grid.dims)

    return new_mask_data_array

