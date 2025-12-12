This repository provides the necessary scripts to reproduce the results and figures from paper "CIMP6 model evaluation of multi-year droughts". 
It contains the following folders and content:

a. Scripts:

	0. Scripts with functions 
	0_functions.py: contains the functions to calculate the multi-year droughts (MYD), the normal droughts (ND), and creates masks for these droughts (mask_MYD and mask_ND). These functions are loaded in in the other scripts where necessary. 
 	0_masks_MYD_ND.py: saved the calculated masks to .nc files
	
	1. Scripts to calculate PET and SPEI 
	1_PenmanMonteith.py: Calculates PET based on Penman-Monteith
	1_Calculate_SPEI.py: Calculates SPEI from PET and PR
 
	2. Scripts for figures
	2_worldmaps_climate.py: Plots different characteristics of multi-year droughts on a global scale. Results in Figure 1 and the global figures in the supplements.
	2_Ratio+distribution.py: Plots the distribution of drought length and the ratio between months in normal drought and months in multi-year drought. Results in Figure 2.
	2_start_times_MYD_ND.py: Plots the distribution of the start times of MYDs and NDs using a bootstrapping method. Results in Figure 3.
	2_drought_probability.py: Plots the probability of being in a normal drought or a multi-year drought based on PET and PR. Results in Figure 4.

b. Masks: 

	Contains masks for central Argentina (ARG), Southeast Australia (AUS), California (CAL), India (IND), Southern Africa (SA), and Western Europe (WEU). All masks are based on (combinations of) river basins and have a resolution of 1x1 degrees.
