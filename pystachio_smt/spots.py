#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" SPOTS - Spot characterisation and manipulation module

Description:
    spots.py contains the Spots class for characterising and manipulating data
    associated with bright spots in the image datasets provided.

Contains:
    class Spots

Author:
    Edward Higgins

Version: 0.2.0
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from algorithms import * # import all functions from alogorithms.py

## Main class for handling spots in frames
class Spots: 
    def __init__(self, num_spots=0, frame=0): #Constructor - Object for spot data in a frame - initilises everything to 0
        self.num_spots = num_spots
        if num_spots > 0:
            self.positions = np.zeros([num_spots, 2])
            self.bg_intensity = np.zeros(num_spots)
            self.spot_intensity = np.zeros(num_spots)
            self.frame = frame
            self.traj_num = [-1] * self.num_spots
            self.snr = np.zeros([num_spots])
            self.laser_on_frame = 0
            self.converged = np.zeros([num_spots], dtype=np.int8)
            self.exists = True
            self.width = np.zeros((num_spots,2))
            self.centre_intensity = np.zeros(num_spots)
        else:
            self.frame = frame
            self.exists = False

    def set_positions(self, positions): # Set the positions of the spots
        self.num_spots = len(positions) # Set the number of spots
        self.positions = np.zeros([self.num_spots, 2]) # initilises the positions
        self.clipping = [False] * self.num_spots # clippping flag
        self.bg_intensity = np.zeros(self.num_spots) # background intensity
        self.spot_intensity = np.zeros(self.num_spots) # spot intensity
        self.width = np.zeros([self.num_spots, 2]) # spot width
        self.traj_num = [-1] * self.num_spots # trajectory number
        self.snr = np.zeros([self.num_spots]) # signal to noise ratio
        self.converged = np.zeros([self.num_spots],dtype=np.int8) # convergence flag
        self.centre_intensity = np.zeros(self.num_spots)


        for i in range(self.num_spots): # Loop through the number of spots
            self.positions[i, :] = positions[i] # Set the positions of the spots

    def find_in_frame(self, frame, params): # Find the spots in the frame
        img_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Convert the frame to BGR

        # Get structural element (map with disk of ones in a square of 0s) [strel] 
        disk_size = 2 * params.struct_disk_radius - 1 # Disk size form radius
        disk_kernel = cv2.getStructuringElement( 
            cv2.MORPH_ELLIPSE, (disk_size, disk_size)
        ) # Create the disk kernel

        # Optionally apply gaussian filtering to the frame
        if params.filter_image == "Gaussian":  # If Gaussian filtering requested
            blurred_frame = cv2.GaussianBlur(img_frame, (3, 3), 0)  # Apply Gaussian blur with 3x3 kernel
        else:  # If no filtering requested
            blurred_frame = img_frame.copy()  # Just copy the original frame

        # Apply top-hat filtering [imtophat]
        tophatted_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, disk_kernel)  # Extract small bright features

        # Get b/w threshold value from the histogram
        hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])  # Calculate histogram
        hist_data[0] = 0  # Zero out first bin (background)

        peak_width, peak_location = fwhm(hist_data)  # Find full-width-half-maximum of histogram
        bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)  # Calculate threshold

        # Apply gaussian filter to the top-hatted image [fspecial, imfilter]
        blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)  # Blur again to reduce noise

        # Convert the filtered image to b/w [im2bw]
        bw_frame = cv2.threshold(  # Threshold the image to binary
            blurred_tophatted_frame, bw_threshold, 255, cv2.THRESH_BINARY
        )[1]

        # "Open" the b/w image (in a morphological sense) [imopen]
        bw_opened = cv2.morphologyEx(  # Perform morphological opening (erosion then dilation)
            bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        )

        # Fill holes of size 1 pixel in the resulting image [bwmorph]
        bw_filled = cv2.morphologyEx(  # Fill small holes by morphological closing
            bw_opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )


        spot_locations = ultimate_erode(bw_filled[:, :, 0], frame) # Find spots w/ ulitmate erode (see algorithms.py)
        if np.isnan(spot_locations).any():
            raise "Found nans"
        self.set_positions(spot_locations)

    def merge_coincident_candidates(self): # Merge coincident candidates
        new_positions = [] # New positions for the merged candidates
        skip = [] # List of spots to skip
        for i in range(self.num_spots): # Loop through the number of spots
            tmp_positions = [self.positions[i, :]] # List of positions to merge
            if i in skip: # If this spot is already merged, skip it
                continue 

            for j in range(i + 1, self.num_spots): # Loop through the number of spots
                if sum((self.positions[i, :] - self.positions[j, :]) ** 2) < 4: # If the distance between the spots is less than 2 pixels
                    skip.append(j) # Add the spot to the skip list
                    tmp_positions.append(self.positions[j, :]) # Add the position to the list of positions to merge

            p = [0, 0] # New position for the merged spot
            for pos in tmp_positions: # Loop through the positions to merge
                p[0] += pos[0] # Add the x position
                p[1] += pos[1] # Add the y position
            p[0] = p[0] / len(tmp_positions) # Average the x position
            p[1] = p[1] / len(tmp_positions) # Average the y position

            new_positions.append(p) # Add the new position to the list of new positions

        self.set_positions(new_positions) # Set the new positions

    def filter_candidates(self, frame, params): # Filter the candidates
        positions = [] # List of positions to keep
        clipping = [] # List of clipping flags
        bg_intensity = [] # List of background intensities
        spot_intensity = [] # List of spot intensities
        centre_intensity = [] # List of centre intensities
        width = [] # List of widths
        traj_num = [] # List of trajectory numbers
        snr = [] # List of signal to noise ratios

        for i in range(self.num_spots): # Loop through the number of spots
            # Fliter spots that are too noisy to be useful candidates
            if self.snr[i] <= params.snr_filter_cutoff: 
                continue
            # Fitler spots that are outside of any existing mask
            if frame.has_mask and frame.mask_data[round(self.positions[i,1]), round(self.positions[i,0])] == 0: 
                continue
            
            # Filter spots too close to the edge to give good numbers ##DWA this code be problematic
            if (
                self.positions[i, 0] < params.subarray_halfwidth
                or self.positions[i, 0] >= frame.frame_size[0] - params.subarray_halfwidth
                or self.positions[i, 1] < params.subarray_halfwidth
                or self.positions[i, 1] >= frame.frame_size[1] - params.subarray_halfwidth
            ):
                continue

            positions.append(self.positions[i, :]) # Add the position to the list of positions to keep
            clipping.append(self.clipping[i]) # Add the clipping flag to the list of clipping flags
            bg_intensity.append(self.bg_intensity[i]) # Add the background intensity to the list of background intensities
            spot_intensity.append(self.spot_intensity[i]) # Add the spot intensity to the list of spot intensities
            centre_intensity.append(self.centre_intensity[i]) # Add the centre intensity to the list of centre intensities
            width.append(self.width[i, :]) # Add the width to the list of widths
            traj_num.append(self.traj_num[i]) # Add the trajectory number to the list of trajectory numbers
            snr.append(self.snr[i]) # Add the signal to noise ratio to the list of signal to noise ratios

        self.num_spots = len(clipping)  # Update number of spots
        self.positions = np.array(positions)  # Update positions array
        self.clipping = np.array(clipping)  # Update clipping array
        self.bg_intensity = np.array(bg_intensity)  # Update background intensity array
        self.spot_intensity = np.array(spot_intensity)  # Update spot intensity array
        self.centre_intensity = np.array(centre_intensity)  # Update center intensity array
        self.width = np.array(width)  # Update width array
        self.traj_num = np.array(traj_num)  # Update trajectory number array
        self.snr = np.array(snr)  # Update SNR array



    def distance_from(self, other):  # Method to calculate distances from another Spots object
        distances = np.zeros([self.num_spots, other.num_spots])  # Initialise distances array

        for i in range(self.num_spots):  # Loop through this object's spots
            for j in range(other.num_spots):  # Loop through other object's spots
                distances[i, j] = np.linalg.norm(  # Calculate Euclidean distance
                    self.positions[i, :] - other.positions[j, :]
                )

        return distances  # Return distances matrix

    def index_first(self):  # Method to number trajectories in first frame
        self.traj_num = list(range(self.num_spots))  # Assign sequential numbers

    def link(self, prev_spots, params):  # Method to link spots with previous frame
        distances = self.distance_from(prev_spots)  # Calculate distances to previous spots

        assigned = []  # Initialise assigned list
        neighbours = np.argsort(distances[:, :], axis=1)  # Sort neighbors by distance
        paired_spots = []  # Initialise paired spots list
        next_trajectory = max(prev_spots.traj_num) + 1  # Initialise next trajectory number
        for i in range(self.num_spots):  # Loop through current spots
            for j in range(prev_spots.num_spots):  # Loop through possible matches
                neighbour = neighbours[i, j]  # Get nearest neighbor
                if distances[i, neighbour] < params.max_displacement:  # If close enough
                    if neighbour in paired_spots:  # If already paired
                        continue  # Skip to next
                    else:  # If not paired
                        paired_spots.append(neighbour)  # Mark as paired
                        self.traj_num[i] = prev_spots.traj_num[neighbour]  # Link trajectories
                else:  # If too far
                    self.traj_num[i] = next_trajectory  # Create new trajectory
                    next_trajectory += 1  # Increment next trajectory number
                break  # Exit inner loop

            if self.traj_num[i] == -1:  # If spot not linked
                sys.exit(f"Unable to find a match for spot {i}, frame {self.frame}")  # Exit with error

    def get_spot_intensities(self, frame, params):  # Method to calculate spot intensities
        for i in range(self.num_spots):  # Loop through all spots
            x = round(self.positions[i, 0])  # Get rounded x coordinate
            y = round(self.positions[i, 1])  # Get rounded y coordinate
            # Create a tmp array with the centre of the spot in the centre
            tmp = frame[  # Extract subarray around spot
                y - params.subarray_halfwidth : y + params.subarray_halfwidth+1, 
                x - params.subarray_halfwidth : x + params.subarray_halfwidth + 1
            ] 
            spotmask = np.zeros(tmp.shape)  # Create empty spot mask
            cv2.circle(spotmask,   # Draw circle on mask
                    (params.subarray_halfwidth, params.subarray_halfwidth),  # Center at middle of subarray
                    params.inner_mask_radius,  # Circle radius
                    1,  # Circle color/value
                    -1  # Fill circle
            )
            bgintensity = np.mean(tmp[spotmask == 0])  # Calculate background as mean outside circle
            tmp = tmp - bgintensity  # Subtract background
            intensity = np.sum(tmp[spotmask == 1])  # Sum intensity within circle
            if intensity == 0:  # If zero intensity
                print(f"WARNING: Zero intensity found at {[x, y]}")  # Print warning
            self.spot_intensity[i] = intensity  # Store spot intensity

    def refine_centres(self, frame, params):  # Method to refine spot centers
        image = frame.as_image()  # Get frame as image
        # Refine the centre of each spot independently
        for i_spot in range(self.num_spots):  # Loop through all spots
            r = params.subarray_halfwidth  # Get subarray halfwidth
            N = 2 * r + 1  # Calculate subarray size

            # Get the centre estimate, make sure the spot_region fits in the frame
            p_estimate = self.positions[i_spot, :]  # Get initial position estimate
            for d in (0, 1):  # For both x and y dimensions
                if round(p_estimate[d]) < r:  # If too close to left/top edge
                    p_estimate[d] = r  # Adjust to fit
                elif round(p_estimate[d]) > frame.frame_size[d]-r-1:  # If too close to right/bottom edge
                    p_estimate[d] = frame.frame_size[d] - r - 1  # Adjust to fit

            # Create the sub-image
            spot_region = np.array(  # Define region around spot
                [
                    [round(p_estimate[0]) - r, round(p_estimate[0]) + r],  # X bounds
                    [round(p_estimate[1]) - r, round(p_estimate[1]) + r],  # Y bounds
                ]
            ).astype(int)

            spot_pixels = image[  # Extract spot region
                spot_region[1, 0] : spot_region[1, 1] + 1,  # Y range
                spot_region[0, 0] : spot_region[0, 1] + 1,  # X range
            ]

            coords = np.mgrid[  # Create coordinate grid
                spot_region[0, 0] : spot_region[0, 1] + 1,  # X coordinates
                spot_region[1, 0] : spot_region[1, 1] + 1,  # Y coordinates
            ]

            Xs, Ys = np.meshgrid(  # Create meshgrid for calculations
                range(spot_region[0, 0], spot_region[0, 1] + 1),  # X range
                range(spot_region[1, 0], spot_region[1, 1] + 1),  # Y range
            )

            converged = False  # Initialise convergence flag
            iteration = 0  # Initialise iteration counter
            clipping = False  # Initialise clipping flag
            spot_intensity = 0  # Initialise spot intensity
            bg_intensity = 0  # Initialise background intensity
            snr = 0  # Initialise SNR
            while not converged and iteration < params.gauss_mask_max_iter:  # Until converged or max iterations
                iteration += 1  # Increment iteration count

                # Generate the inner mask
                inner_mask = np.where(  # Create circular mask
                    (coords[0, :, :] - p_estimate[0]) ** 2  # X distance squared
                    + (coords[1, :, :] - p_estimate[1]) ** 2  # Y distance squared
                    <= params.inner_mask_radius ** 2,  # Compare to radius squared
                    1,  # Inside circle
                    0,  # Outside circle
                )
                mask_pixels = np.sum(inner_mask)  # Count pixels in mask

                # Generate the Gaussian mask
                # This uses Numpy magic, it's almost as bad as the MATLAB...
                coords_sq = (  # Calculate squared distances
                    coords[:, :, :] - p_estimate[:, np.newaxis, np.newaxis]  # Subtract center
                ) ** 2  # Square
                exponent = -(coords_sq[0, :, :] + coords_sq[1, :, :]) / (  # Negative sum of squares
                    2 * params.gauss_mask_sigma ** 2  # Divided by 2*sigma²
                )
                gauss_mask = np.exp(exponent)  # Gaussian = e^(-r²/2σ²)

                if np.sum(gauss_mask) != 0:  # If mask has nonzero sum
                    gauss_mask /= np.sum(gauss_mask)  # Normalise mask

                bg_mask = 1 - inner_mask  # Background mask is inverse of inner mask

                # Calculate the local background intensity and subtract it off the sub-image
                spot_bg = spot_pixels * bg_mask  # Get background pixels
                num_bg_spots = np.sum(bg_mask)  # Count background pixels
                bg_average = np.sum(spot_bg) / num_bg_spots  # Calculate average background

                # Calculate background corrected sub-image
                bg_corr_spot_pixels = spot_pixels - bg_average  # Subtract background

                # Calculate revised position estimate
                spot_gaussian_product = bg_corr_spot_pixels * gauss_mask  # Weight pixels by Gaussian
                p_estimate_new = np.zeros(2)  # Initialise new estimate
                p_estimate_new[0] = np.sum(spot_gaussian_product * Xs) / np.sum(  # Weighted average X
                    spot_gaussian_product
                )
                p_estimate_new[1] = np.sum(spot_gaussian_product * Ys) / np.sum(  # Weighted average Y
                    spot_gaussian_product
                )
                estimate_change = np.linalg.norm(p_estimate - p_estimate_new)  # Calculate change in estimate

                if not np.isnan(p_estimate_new).any():  # If no NaNs in new estimate
                    p_estimate = p_estimate_new  # Update estimate
                else:  # If NaNs found
                    print("WARNING: Position estimate is NaN, falied to converge")  # Print warning
                    break  # Exit loop

                spot_intensity = np.sum(bg_corr_spot_pixels * inner_mask)  # Calculate spot intensity
                bg_std = np.std(spot_bg[bg_mask==1])  # Calculate background standard deviation


                if estimate_change < 1e-6:  # If position change is small
                    converged = True  # Mark as converged

                # Calculate signal-noise ratio
                # Don't bother reiterating this spot if it's too low
                snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))  # Calculate SNR
#EJH#                 snr = abs(spot_intensity / (bg_std*np.sum(inner_mask)))  # Commented duplicate line
                if snr <= params.snr_filter_cutoff:  # If SNR below cutoff
                    break  # Exit loop

            self.bg_intensity[i_spot] = bg_average  # Store background intensity
            self.spot_intensity[i_spot] = spot_intensity  # Store spot intensity
            self.snr[i_spot] = snr  # Store SNR
            self.converged[i_spot] = converged  # Store convergence flag

            self.positions[i_spot, :] = p_estimate  # Update position with refined estimate
            
    def get_spot_widths(self, frame, params):  # Method to calculate spot widths
        for i in range(self.num_spots):  # Loop through all spots
            x = round(self.positions[i, 0])  # Get rounded x coordinate
            y = round(self.positions[i, 1])  # Get rounded y coordinate
            # Create a tmp array with the centre of the spot in the centre
            tmp = frame[  # Extract subarray around spot
                y - params.subarray_halfwidth : y + params.subarray_halfwidth+1, 
                x - params.subarray_halfwidth : x + params.subarray_halfwidth + 1
                ] 
            spotmask = np.zeros(tmp.shape)  # Create empty mask
            cv2.circle(spotmask,   # Draw circle on mask
             (params.subarray_halfwidth, params.subarray_halfwidth),  # Center of circle
             params.inner_mask_radius,  # Circle radius
             1,  # Circle color/value
             -1  # Fill circle
             )
            bgintensity = np.mean(tmp[spotmask == 0])  # Calculate background intensity
            tmp = tmp - bgintensity  # Subtract background
            p, succ = fit2Dgaussian(tmp)  # Fit 2D Gaussian to spot
            if succ==1: # the fit is OK  # If fit successful
                self.width[i,0] = p[3]  # Store x width
                self.width[i,1] = p[4]  # Store y width
            else: # something went wrong  # If fit failed
                self.width[i,0] = params.psf_width  # Use default width
                self.width[i,1] = params.psf_width  # Use default width