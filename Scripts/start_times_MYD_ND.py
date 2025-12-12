#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:49:44 2025

@author: Jonna van Mourik
Start times MYDs and NDs
"""
#Import things
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append("/your/dir/")
from functions import MYD, ND
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib import cm

#%% Calculate when MYDs and NDs start per model
dir = "/your/dir/"
models = ["MPI-ESM1-2-LR", "CanESM5", "CESM2", "EC-Earth3", "ACCESS-ESM1-5", "MIROC6"]
regions = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]

nr_MYD_DJF_models = []
nr_MYD_MAM_models = []
nr_MYD_JJA_models = []
nr_MYD_SON_models = []

nr_ND_DJF_models = []
nr_ND_MAM_models = []
nr_ND_JJA_models = []
nr_ND_SON_models = []

for model in models:
    dir_mask = dir+"masks/"+str(model)+"/"
    mask_MYD = [xr.open_dataarray(dir_mask+f"mask_MYD_{model}_{region}_1850-2014.nc") for region in regions]
    mask_ND = [xr.open_dataarray(dir_mask+f"mask_ND_{model}_{region}_1850-2014.nc") for region in regions]
    #Precompute season assignment for all masks
    start_MYD = [mask & ~mask.shift(time=1, fill_value=False) for mask in mask_MYD]
    start_MYD = [start.assign_coords(season=start["time"].dt.season) for start in start_MYD]
    
    start_ND = [mask & ~mask.shift(time=1, fill_value=False) for mask in mask_ND]
    start_ND = [start.assign_coords(season=start["time"].dt.season) for start in start_ND]
    #Group by season and compute drought counts in one step
    nr_MYD_DJF_regions = []
    nr_MYD_MAM_regions = []
    nr_MYD_JJA_regions = []
    nr_MYD_SON_regions = []
    
    nr_ND_DJF_regions = []
    nr_ND_MAM_regions = []
    nr_ND_JJA_regions = []
    nr_ND_SON_regions = []
    # Process each region
    for start in start_ND:
    # Group by season and count droughts
        season_counts = start.groupby("season").sum(dim="time")

        # Extract drought counts for each season
        nr_ND_DJF_regions.append(season_counts.sel(season="DJF", drop=True).values)
        nr_ND_MAM_regions.append(season_counts.sel(season="MAM", drop=True).values)
        nr_ND_JJA_regions.append(season_counts.sel(season="JJA", drop=True).values)
        nr_ND_SON_regions.append(season_counts.sel(season="SON", drop=True).values)

    # Append regional results for the current model
    nr_ND_DJF_models.append([np.sum(region) for region in nr_ND_DJF_regions])
    nr_ND_MAM_models.append([np.sum(region) for region in nr_ND_MAM_regions])
    nr_ND_JJA_models.append([np.sum(region) for region in nr_ND_JJA_regions])
    nr_ND_SON_models.append([np.sum(region) for region in nr_ND_SON_regions])
    
    for start in start_MYD:
    # Group by season aMYD count droughts
        season_counts = start.groupby("season").sum(dim="time")

        # Extract drought counts for each season
        nr_MYD_DJF_regions.append(season_counts.sel(season="DJF", drop=True).values)
        nr_MYD_MAM_regions.append(season_counts.sel(season="MAM", drop=True).values)
        nr_MYD_JJA_regions.append(season_counts.sel(season="JJA", drop=True).values)
        nr_MYD_SON_regions.append(season_counts.sel(season="SON", drop=True).values)

    # append regional results for the current model
    nr_MYD_DJF_models.append([np.sum(region) for region in nr_MYD_DJF_regions])
    nr_MYD_MAM_models.append([np.sum(region) for region in nr_MYD_MAM_regions])
    nr_MYD_JJA_models.append([np.sum(region) for region in nr_MYD_JJA_regions])
    nr_MYD_SON_models.append([np.sum(region) for region in nr_MYD_SON_regions])
    

#%% Bootstrapping method
# Perform bootstrapping
#from scipy.stats import bootstrap
n_bootstrap = 1000  # Number of bootstrap iterations
size = 100

def bootstrap(dataarray, n_samples, size):
    bootstrapped_samples = []
    for _ in range(n_samples):
        # Get flat indices for sampling
        indices = np.random.choice(dataarray.sample.size, size=size, replace=True)
        
        # Select the resampled values
        resampled = dataarray.isel(sample=indices)
        bootstrapped_samples.append(resampled)
    return bootstrapped_samples

#Combine with original code
models = ["MPI-ESM1-2-LR", "CanESM5", "CESM2", "EC-Earth3", "ACCESS-ESM1-5", "MIROC6"]
regions = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]

nr_MYD_DJF_bs = []
nr_MYD_MAM_bs = []
nr_MYD_JJA_bs = []
nr_MYD_SON_bs = []

nr_ND_DJF_bs = []
nr_ND_MAM_bs = []
nr_ND_JJA_bs = []
nr_ND_SON_bs = []

std_MYD_DJF_bs = []
std_MYD_MAM_bs = []
std_MYD_JJA_bs = []
std_MYD_SON_bs = []

std_ND_DJF_bs = []
std_ND_MAM_bs = []
std_ND_JJA_bs = []
std_ND_SON_bs = []

for model in models:
    dir_mask = "/masks/"+str(model)+"/"
    mask_MYD = [xr.open_dataarray(dir_mask+f"mask_MYD_{model}_{region}_1850-2014.nc") for region in regions]
    mask_ND = [xr.open_dataarray(dir_mask+f"mask_ND_{model}_{region}_1850-2014.nc") for region in regions]
    #Precompute season assignment for all masks
    start_MYD = [mask & ~mask.shift(time=1, fill_value=False) for mask in mask_MYD]
    start_MYD = [start.assign_coords(season=start["time"].dt.season) for start in start_MYD]
    
    start_MYD_true = [start.where(start, drop=True).stack(sample=("member", "time")).dropna(dim="sample") for start in start_MYD]
    
    start_ND = [mask & ~mask.shift(time=1, fill_value=False) for mask in mask_ND]
    start_ND = [start.assign_coords(season=start["time"].dt.season) for start in start_ND]
    
    start_ND_true = [start.where(start, drop=True).stack(sample=("member", "time")).dropna(dim="sample") for start in start_ND]
    #Group by season and compute drought counts in one step
    nr_MYD_DJF_regions = []
    nr_MYD_MAM_regions = []
    nr_MYD_JJA_regions = []
    nr_MYD_SON_regions = []
    
    nr_ND_DJF_regions = []
    nr_ND_MAM_regions = []
    nr_ND_JJA_regions = []
    nr_ND_SON_regions = []
    # Process each region
    for start in start_ND_true:
    # Group by season and count droughts
        
        bootstrap_results = bootstrap(start, n_bootstrap, size)
        djf_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "DJF").count().values
            for sample in bootstrap_results]
        mam_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "MAM").count().values
            for sample in bootstrap_results]
        jja_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "JJA").count().values
            for sample in bootstrap_results]
        son_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "SON").count().values
            for sample in bootstrap_results]

        # Extract drought counts for each season
        nr_ND_DJF_regions.append(djf_counts)
        nr_ND_MAM_regions.append(mam_counts)
        nr_ND_JJA_regions.append(jja_counts)
        nr_ND_SON_regions.append(son_counts)
    
    # Append regional results for the current model
    nr_ND_DJF_bs.append([np.mean(region)/size*100 for region in nr_ND_DJF_regions])
    nr_ND_MAM_bs.append([np.mean(region)/size*100 for region in nr_ND_MAM_regions])
    nr_ND_JJA_bs.append([np.mean(region)/size*100 for region in nr_ND_JJA_regions])
    nr_ND_SON_bs.append([np.mean(region)/size*100 for region in nr_ND_SON_regions])
    
    std_ND_DJF_bs.append([np.std(region)/size*100 for region in nr_ND_DJF_regions])
    std_ND_MAM_bs.append([np.std(region)/size*100 for region in nr_ND_MAM_regions])
    std_ND_JJA_bs.append([np.std(region)/size*100 for region in nr_ND_JJA_regions])
    std_ND_SON_bs.append([np.std(region)/size*100 for region in nr_ND_SON_regions])
    
    for start in start_MYD_true:
    # Group by season aMYD count droughts
        bootstrap_results = bootstrap(start, n_bootstrap, size=size)
        djf_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "DJF").count().values
            for sample in bootstrap_results]
        mam_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "MAM").count().values
            for sample in bootstrap_results]
        jja_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "JJA").count().values
            for sample in bootstrap_results]
        son_counts = [
            sample["time"].dt.season.where(sample["time"].dt.season == "SON").count().values
            for sample in bootstrap_results]
        # Extract drought counts for each season
        nr_MYD_DJF_regions.append(djf_counts)
        nr_MYD_MAM_regions.append(mam_counts)
        nr_MYD_JJA_regions.append(jja_counts)
        nr_MYD_SON_regions.append(son_counts)

    # append regional results for the current model
    nr_MYD_DJF_bs.append([np.mean(region)/size*100 for region in nr_MYD_DJF_regions])
    nr_MYD_MAM_bs.append([np.mean(region)/size*100 for region in nr_MYD_MAM_regions])
    nr_MYD_JJA_bs.append([np.mean(region)/size*100 for region in nr_MYD_JJA_regions])
    nr_MYD_SON_bs.append([np.mean(region)/size*100 for region in nr_MYD_SON_regions])
    
    std_MYD_DJF_bs.append([np.std(region)/size*100 for region in nr_MYD_DJF_regions])
    std_MYD_MAM_bs.append([np.std(region)/size*100 for region in nr_MYD_MAM_regions])
    std_MYD_JJA_bs.append([np.std(region)/size*100 for region in nr_MYD_JJA_regions])
    std_MYD_SON_bs.append([np.std(region)/size*100 for region in nr_MYD_SON_regions])

# Make new version of the percentages figure
region_names = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]

#%% Alternative with one marker but in different shades of brown and green

region_names = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
models = ["MPI-ESM1-2-LR", "CanESM5", "CESM2", "EC-Earth3", "ACCESS-ESM1-5", "MIROC6"]

# Custom brown colormap (shades of brown from RGB manually or define your own)
from matplotlib.colors import ListedColormap
brown_shades = ListedColormap(["#8B4513", "#A0522D", "#CD853F", "#D2B48C", "#DEB887", "#F5DEB3"])

# Color palettes
MYD_colors = brown_shades(np.linspace(0, 0.8, len(models)))
ND_colors = cm.Greens(np.linspace(0.9, 0.45, len(models)))  # Adjust bounds for visual separation

# Plot setup
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True)
axes = axes.flatten()
fontsize_labels = 18
bar_width = 0.12

for r, ax in enumerate(axes):
    region_index = r
    region_name = region_names[r]

    MYD_data = {season: [model_data[region_index] for model_data in dataset] 
                for season, dataset in zip(["DJF", "MAM", "JJA", "SON"], 
                                           [nr_MYD_DJF_bs, nr_MYD_MAM_bs, nr_MYD_JJA_bs, nr_MYD_SON_bs])}
    
    ND_data = {season: [model_data[region_index] for model_data in dataset] 
               for season, dataset in zip(["DJF", "MAM", "JJA", "SON"], 
                                          [nr_ND_DJF_bs, nr_ND_MAM_bs, nr_ND_JJA_bs, nr_ND_SON_bs])}

    MYD_std = {season: [model_data[region_index] for model_data in dataset] 
               for season, dataset in zip(["DJF", "MAM", "JJA", "SON"], 
                                          [std_MYD_DJF_bs, std_MYD_MAM_bs, std_MYD_JJA_bs, std_MYD_SON_bs])}
    
    ND_std = {season: [model_data[region_index] for model_data in dataset] 
              for season, dataset in zip(["DJF", "MAM", "JJA", "SON"], 
                                         [std_ND_DJF_bs, std_ND_MAM_bs, std_ND_JJA_bs, std_ND_SON_bs])}

    seasons = list(MYD_data.keys())
    x = np.arange(len(seasons))

    for i, model in enumerate(models):
        y_values_MYD = [MYD_data[season][i] for season in seasons]
        y_std_MYD = [MYD_std[season][i] for season in seasons]
        y_values_ND = [ND_data[season][i] for season in seasons]
        y_std_ND = [ND_std[season][i] for season in seasons]

        ax.errorbar(x + i * bar_width, y_values_MYD, yerr=y_std_MYD, fmt='o', 
                    color=MYD_colors[i], markersize=4, zorder=5)
        
        ax.errorbar(x + i * bar_width, y_values_ND, yerr=y_std_ND, fmt='o', 
                    color=ND_colors[i], markersize=4, zorder=4)

    ax.set_title(f"{region_name}", fontsize=fontsize_labels)
    if r in [3, 4, 5]:
        ax.set_xlabel("Season", fontsize=fontsize_labels-2)
    if r in [0, 3]:
        ax.set_ylabel("Percentage [%]", fontsize=fontsize_labels-2)

    ax.set_xticks(x + (len(models) - 1) * bar_width / 2)
    ax.set_xticklabels(seasons, fontsize=fontsize_labels-4)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis='y', labelsize=fontsize_labels-4)
    ax.set_ylim(0, 80)

# Legend with paired color dots
legend_elements = []
for j in range(len(models)):
    myd_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor=MYD_colors[j], markersize=6)
    nd_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor=ND_colors[j], markersize=6)
    legend_elements.append((myd_marker, nd_marker))

fig.legend(
    legend_elements, models,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    bbox_to_anchor=(1.0, 0.95), loc="upper left", ncol=1, fontsize=fontsize_labels-2)

fig.text(1.01, 0.95, "MYDs |", color="brown", fontsize=fontsize_labels+2)
fig.text(1.11, 0.95, "NDs", color="green", fontsize=fontsize_labels+2)

plt.tight_layout()
plt.show()
fig.savefig(f"Percentage_of_MYDs_per_season_bootstrap_sample={size}_replace=true_n={n_bootstrap}_v5.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"Percentage_of_MYDs_per_season_bootstrap_sample={size}_replace=true_n={n_bootstrap}_v5.pdf", bbox_inches="tight")
