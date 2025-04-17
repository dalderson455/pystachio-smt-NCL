# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>

# Distributed under terms of the MIT license.

""" ALGORITHMS - Low level algorithms module

Description:
    algorithms.py contains a number of useful algorithms that are used
    throughout the code, but don't necessarily need any of the data structures
    defined in other modules.

Contains:
    function fwhm
    function find_local_maxima_scipy
    function ultimate_erode_scipy
    function gaussian_formula
    function gaussian_2d_for_curve_fit
    function moments
    function fit2Dgaussian

Author:
    Edward Higgins & JWS
    
    DWA - 2023-10-03 - Added scipy.ndimage functions to replace numba.jit
    Gemini - 2025-04-14 - Refreshed comments

Version: 0.2.1
"""

import sys

#import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy import optimize
from scipy.optimize import curve_fit

def fwhm(data):
    # Create array of indices (0-255)
    x = np.linspace(0, 255, 256).astype(int)

    # Normalise data (max value becomes 1)
    data = data / np.max(data)
    # Get size of data array minus 1
    N = data.size - 1

    # Define half-max level
    lev50 = 0.5
    # Check if peak is positive or negative
    if data[0] < lev50:
        # Find index of max value
        centre_index = np.argmax(data)
        # Polarity flag (positive peak)
        Pol = +1
    else:
        # Find index of min value
        centre_index = np.argmin(data)
        # Polarity flag (negative peak)
        Pol = -1

    # Handle case where peak is at edge
    if centre_index > 254:
        print("WARNING: Struggling to find a peak in histogram")
        centre_index = 254

    # Get index value of peak centre
    extremum_val = x[centre_index]

    # Find leading edge crossing 50% level
    i = 1
    # Iterate until sign change or end of array
    while np.sign(data[i] - lev50) == np.sign(data[i - 1] - lev50) and i < 255:
        i += 1

    # Interpolate exact crossing point (leading edge)
    interp = (lev50 - data[i - 1]) / (data[i] - data[i - 1])
    lead_t = x[i - 1] + interp * (x[i] - x[i - 1])

    # Find trailing edge crossing 50% level
    i = centre_index + 1
    # Iterate from peak until sign change or end of array
    while (np.sign(data[i] - lev50) == np.sign(data[i - 1] - lev50)) and (i <= N - 1):
        i += 1

    # Check if trailing edge found within array
    if i != N:
        # Interpolate exact crossing point (trailing edge)
        interp = (lev50 - data[i - 1]) / (data[i] - data[i - 1])
        trail_t = x[i - 1] + interp * (x[i] - x[i - 1])
        # Calculate FWHM
        x_width = trail_t - lead_t
    else:
        # Peak goes off edge, width is zero
        trail_t = None
        x_width = 0

    # Return width and peak position
    return (x_width, extremum_val)

def find_local_maxima_scipy(img):
    # Return empty if input image is invalid
    if img is None or img.size == 0:
        return []
        
    # Find max value in 3x3 neighbourhood for each pixel
    maximum_img = scipy.ndimage.maximum_filter(img, size=3, mode='constant', cval=0.0)

    # Create mask where pixel value equals neighbourhood max
    local_max_mask = (img == maximum_img)

    # Ensure maxima are also non-zero
    local_max_mask &= (img > 0)

    # Get coordinates (y, x) of maxima from mask
    coords_yx = np.nonzero(local_max_mask)

    # Convert (y, x) coordinates to list of [x, y]
    if coords_yx[0].size > 0:
        # Stack x and y coords, transpose, convert to list
        local_maxima_xy = np.stack((coords_yx[1], coords_yx[0]), axis=-1).tolist()
    else:
        # Return empty list if no maxima found
        local_maxima_xy = []

    # Return list of [x, y] maxima coordinates
    return local_maxima_xy

def ultimate_erode_scipy(img):
    # Return empty if input image is invalid or all zeros
    if img is None or img.size == 0 or np.all(img == 0):
        return []

    # Calc Euclidean Distance Transform (dist to nearest zero pixel)
    img_dist = scipy.ndimage.distance_transform_edt(img)

    # Find peaks (local maxima) in the distance map
    # These peaks represent object centres (ultimate eroded points)
    spot_locations = find_local_maxima_scipy(img_dist)

    # Return list of [x, y] centre coordinates
    return spot_locations

def gaussian_formula(x, y, height, center_x, center_y, width_x, width_y):
    # Ensure widths are floating point numbers
    width_x = float(width_x)
    width_y = float(width_y)
    
    # Prevent division by zero or log(neg) errors; ensure widths > small epsilon
    width_x = max(width_x, 1e-6)
    width_y = max(width_y, 1e-6)
    
    # Calculate distances from centre
    xp = center_x - x
    yp = center_y - y
    
    # Calculate exponent term for 2D Gaussian
    exponent = -((xp / width_x)**2 + (yp / width_y)**2) / 2
    
    # Calculate Gaussian value
    return height * np.exp(exponent)

def gaussian_2d_for_curve_fit(coords, height, center_x, center_y, width_x, width_y):
    # Unpack flattened x and y coordinate arrays
    x, y = coords
    
    # Calculate Gaussian values for all coordinates using the formula
    g = gaussian_formula(x, y, height, center_x, center_y, width_x, width_y)
    
    # Return flattened array of Gaussian values for curve_fit
    return g.ravel()

def moments(data):
    # Calculate total intensity sum
    total = data.sum()
    
    # Check for invalid data (empty, zero sum, too small)
    if total <= 0 or data.shape[0] < 2 or data.shape[1] < 2:
         # Return default guess if data is bad
         return 1000, data.shape[1]//2, data.shape[0]//2, 1.5, 1.5

    # Get Y, X index arrays matching data shape
    Y, X = np.indices(data.shape)
    
    # Calculate intensity-weighted centroid (y, x)
    y = (Y*data).sum()/total
    x = (X*data).sum()/total

    # Ensure centroid indices are within array bounds
    x_idx = int(np.clip(x, 0, data.shape[1]-1))
    y_idx = int(np.clip(y, 0, data.shape[0]-1))

    # Estimate width_x (std dev) along row through centroid
    row = data[y_idx, :]
    # Check row sum to avoid errors
    if row.sum() > 0:
         # Calculate intensity-weighted standard deviation for x
         width_x = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    else:
         # Use fallback width if row sum is zero
         width_x = 1.5

    # Estimate width_y (std dev) along column through centroid
    col = data[:, x_idx]
    # Check column sum to avoid errors
    if col.sum() > 0:
         # Calculate intensity-weighted standard deviation for y
         width_y = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    else:
         # Use fallback width if column sum is zero
         width_y = 1.5

    # Estimate height from pixel value at centroid
    height = data[y_idx, x_idx]
    
    # Ensure calculated widths are positive
    width_x = max(width_x, 1e-6)
    width_y = max(width_y, 1e-6)

    # Return estimated Gaussian parameters: height, x, y, width_x, width_y
    return height, x, y, width_x, width_y

def fit2Dgaussian(data):
    # Calculate initial guess for Gaussian parameters using moments
    params_initial = moments(data) # Order: [height, x, y, wx, wy]

    # Create grid of Y and X indices matching data shape
    Y_indices, X_indices = np.indices(data.shape)

    # Flatten coordinate arrays and data array for curve_fit
    x_data_flat = X_indices.ravel()
    y_data_flat = Y_indices.ravel()
    # Pass coordinates as a tuple (required format for curve_fit func)
    coords = (x_data_flat, y_data_flat)
    data_flat = data.ravel()

    # Initialise success flag and optimal parameters (popt)
    success = 0
    popt = params_initial # Default to initial guess

    try:
        # Use scipy.optimize.curve_fit to find best Gaussian parameters
        # func: the Gaussian model function
        # xdata: the flattened coordinates tuple
        # ydata: the flattened pixel intensities
        # p0: the initial parameter guess
        popt, pcov = curve_fit(
            gaussian_2d_for_curve_fit,
            coords,
            data_flat,
            p0=params_initial
        )
        # If fit succeeds without error, set success flag
        success = 1

    except RuntimeError:
        # Handle cases where the fit does not converge
        #print("Warning: 2D Gaussian fit did not converge, using initial guess.")
        # success remains 0, popt remains initial guess
        pass
    except Exception as e:
        # Handle any other unexpected errors during fitting
        #print(f"Warning: An error occurred during 2D Gaussian fit: {e}")
        # success remains 0, popt remains initial guess
        pass

    # Return optimal parameters found and success flag
    return popt, success