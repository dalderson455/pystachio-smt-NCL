#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" TRACKING - Spot tracking module

Description:
    tracking.py contains the code for the tracking task, which identifies spots
    within a set of frames and builds spot trajectories across those frames.

Contains:
    function track

Author:
    Edward Higgins

Version: 0.2.0
"""

import sys

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import spots
import trajectories
import images


def track(params):
    # Read in the image data
    image_data = images.ImageData()
    image_data.read(params.name + ".tif", params)
    ### ALEX stuff ###
    if params.ALEX==True:
        imageL=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
        imageR=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
        if params.start_channel=='L':
            for i in range(0,image_data.num_frames-1,2):
                imageL[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
                imageR[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
        else:
            for i in range(0,image_data.num_frames-1,2):
                imageR[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
                imageL[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
        image_data.num_frames = image_data.num_frames//2

        #LHS
        image_data.pixel_data = imageL
        image_data.frame_size = [image_data.frame_size[0]//2,image_data.frame_size[1]]
        all_spots = []
        for frame in range(image_data.num_frames):
            all_spots.append(track_frame(image_data[frame], frame, params))
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name +  "_Lchannel_trajectories.csv")

        #RHS
        image_data.pixel_data = imageR
        all_spots = []
        for frame in range(image_data.num_frames):
            all_spots.append(track_frame(image_data[frame], frame, params))
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name +  "_Rchannel_trajectories.csv")

    # For each frame, detect spots NOT ALEX
    else:
        all_spots = [] # Empty list to hold the spot data for each frame
        if params.num_procs == 0: # No multiprocessing  
            # Loop over the frames and find the spots
            for frame in range(image_data.num_frames): # Loop over the frames
                all_spots.append(track_frame(image_data[frame], frame, params)) # Find spots in this frame

        else: # Multiprocessing
            res = [None] * image_data.num_frames # Create results array
            with mp.Pool(params.num_procs) as pool: # Create process pool
                for frame in range(image_data.num_frames): #Loop through frames
                    res[frame] = pool.apply_async(track_frame, (image_data[frame], frame, params)) # Find spots in this frame
                for frame in range(image_data.num_frames): # Loop through frames
                    all_spots.append(res[frame].get()) # Get the results from the processes

        # Link the spot trajectories across the frames
        trajs = trajectories.build_trajectories(all_spots, params) # Build the trajectories
        trajectories.write_trajectories(trajs, params.name + "_trajectories.csv") # Write the trajectories to a file

def track_frame(frame_data, frame, params): # Function to find the spots in a frame
        frame_spots = spots.Spots(frame=frame) # Create a new Spots object for this frame
        frame_spots.find_in_frame(frame_data.as_image()[:, :], params) # Find the spots in this frame
        found_spots = frame_spots.num_spots # Number of spots found in this frame
        frame_spots.merge_coincident_candidates() # Merge coincident candidates

        merged_spots = frame_spots.num_spots
        # Iteratively refine the spot centres
        frame_spots.refine_centres(frame_data, params) # Refine the spot centres
        frame_spots.filter_candidates(frame_data, params) # Filter the candidates
        frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params) # Get the spot intensities
        frame_spots.get_spot_widths(frame_data.as_image()[:,:], params) # Get the spot widths
        if params.verbose: # If Verbose(?) is enabled
            print(
                f"Frame {frame:4d}: found {frame_spots.num_spots:3d} spots "
                f"({found_spots:3d} identified, "
                f"{found_spots-merged_spots:3d} merged, "
                f"{merged_spots-frame_spots.num_spots:3d} filtered)"
            )  # print spot stats
        return frame_spots # Return the spots found in this frame
