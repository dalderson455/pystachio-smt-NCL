# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>

# Distributed under terms of the MIT license.

""" SIMULATION - Dataset simulation module

Description:
    simulation.py contains the code for the simulation task, which simulates
    pseudo-experimental datasets as characterised by the relevant parameters.

Contains:
    function simulate

Author:
    Edward Higgins

Version: 0.2.0
"""

# --- Core library imports ---
from functools import reduce
import numpy as np
import numpy.random as random
import sys

# --- Optional imports (commented out) ---
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt

# --- Local module imports ---
from . import images
from .images import ImageData
from . import spots
from .spots import Spots
from . import trajectories

# --- Main function to simulate image data and trajectories ---
def simulate(params):
    # Validate number of frames parameter
    if params.num_frames < 1:
        sys.exit("ERROR: Cannot simulate image with num_frames < 0")

    # Initialise list of Spots objects for ground truth data
    real_spots = [Spots(params.num_spots) for i in range(params.num_frames)]
    # Determine initial molecule count per spot
    if params.max_spot_molecules == 1:
        n_mols = np.array([1] * params.num_spots) # Fixed count
    else:
        # Random count between 1 and max
        n_mols = np.array(random.randint(1, params.max_spot_molecules, params.num_spots))
    # Array for fractional intensity loss due to intra-frame bleaching
    n_mols_fractional_intensity = np.zeros(n_mols.shape)

    # Initialise first frame (frame 0) spot positions and intensity
    real_spots[0].positions[:, 0] = random.rand(params.num_spots) * params.frame_size[0] # Random x
    real_spots[0].positions[:, 1] = random.rand(params.num_spots) * params.frame_size[1] # Random y
    real_spots[0].spot_intensity[:] = params.I_single # Initial intensity
    real_spots[0].frame = 0 # Set frame number (adjusting if needed later, seems inconsistent with loop start)

    # Calculate diffusion step size in pixels
    S = np.sqrt(2 * params.diffusion_coeff * params.frame_time) / params.pixel_size

    # --- Simulate subsequent frames (diffusion and bleaching) ---
    for frame in range(1, params.num_frames):
        # Set frame number for current Spots object
        real_spots[frame].frame = frame
        # Calculate intensity based on remaining molecules + fractional loss from previous frame
        real_spots[frame].spot_intensity[:] = params.I_single * (n_mols + n_mols_fractional_intensity)
        # Propagate trajectory numbers (assuming fixed spot identities for simulation)
        real_spots[frame].traj_num = real_spots[frame - 1].traj_num[:] # This needs init for frame 0
        # Simulate diffusion step from previous frame's position
        real_spots[frame].positions = random.normal(
            real_spots[frame - 1].positions, S, (params.num_spots, 2)
        )

        # --- Simulate Photobleaching for the *next* frame ---
        n_mols_fractional_intensity[:] = 0 # Reset fractional loss accumulator
        for i in range(params.num_spots):
            # Process only spots with remaining molecules
            if n_mols[i] > 0:
                molecules_to_bleach = 0
                # Check each molecule for bleaching event
                for j in range(n_mols[i]):
                    if random.rand() < params.p_bleach_per_frame:
                        molecules_to_bleach += 1
                        # Calculate fractional survival time within the frame
                        frac = random.rand()
                        # Add fractional intensity contribution for the *next* frame
                        n_mols_fractional_intensity[i] += frac
                # Update molecule count for the *next* frame
                n_mols[i] -= molecules_to_bleach
        # --- End Photobleaching Simulation ---

    # --- Generate Image Stack from Simulated Spots ---
    # Initialise ImageData object
    image = ImageData()
    image.initialise(params.num_frames, params.frame_size)

    # Create coordinate grid matching frame size
    x_pos, y_pos = np.meshgrid(range(params.frame_size[0]), range(params.frame_size[1]))
    # Generate pixel data for each frame
    for frame in range(params.num_frames):
        # Initialise frame with zeros
        frame_data = np.zeros([params.frame_size[1], params.frame_size[0]]).astype(np.uint16)

        # Add intensity contribution from each spot
        for spot in range(params.num_spots):
            spot_intensity_value = real_spots[frame].spot_intensity[spot]
            # Add Gaussian profile only if spot is 'on' (intensity > 0)
            if spot_intensity_value > 0:
                # Calculate Gaussian intensity profile centered at spot position
                spot_data = (
                    (spot_intensity_value / (2 * np.pi * params.spot_width**2)) # Normalisation
                    * np.exp(
                        -(
                            (x_pos - real_spots[frame].positions[spot, 0]) ** 2
                            + (y_pos - real_spots[frame].positions[spot, 1]) ** 2
                        )
                        / (2 * params.spot_width ** 2) # Gaussian width from params
                    )
                ).astype(np.uint16) # Convert profile to uint16
                # Add spot profile to frame data
                frame_data += spot_data
                # Update spot intensity field (redefining it as summed intensity?)
                real_spots[frame].spot_intensity[spot]=np.sum(spot_data)
            else: # If spot bleached, ensure stored intensity is zero
                 real_spots[frame].spot_intensity[spot] = 0

        # --- Add Noise to the Frame ---
        # Add Poisson noise (shot noise)
        frame_data = random.poisson(frame_data).astype(np.uint16)
        # Add Gaussian background noise (read noise, etc.)
        bg_noise = random.normal(params.bg_mean, params.bg_std, [params.frame_size[1], params.frame_size[0]])
        # Add background, ensuring non-negativity
        frame_data += np.where(bg_noise > 0, bg_noise.astype(np.uint16), 0)
        # --- End Noise Addition ---

        # Assign generated frame data to the ImageData object
        image[frame] = frame_data

    # --- Build and Save Simulated Trajectories ---
    # Build trajectories from the simulated ground truth spots
    # Need to initialise traj_num in real_spots[0] first for this to work
    real_spots[0].index_first() # Assign initial trajectory numbers (0 to N-1)
    real_trajs = trajectories.build_trajectories(real_spots, params)

    # Write simulated image stack and trajectory data to files
    image.write(params.name + ".tif")
    trajectories.write_trajectories(real_trajs, params.name + '_simulated.csv')

    # Return simulated data
    return image, real_trajs

# """ SIMULATION - Dataset simulation module

# Description:
#     simulation.py contains the code for the simulation task, which simulates
#     pseudo-experimental datasets as characterised by the relevant parameters.

# Contains:
#     function simulate

# Author:
#     Edward Higgins

# Version: 0.2.0
# """

# from functools import reduce

# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# import numpy as np
# import numpy.random as random
# import sys
# # Import local modules
# from . import images
# from .images import ImageData
# from . import spots
# from .spots import Spots # Explicitly import Spots class
# from . import trajectories # Explicitly import trajectories module

# def simulate(params):
#     # Check for valid number of frames
#     if params.num_frames < 1:
#         sys.exit("ERROR: Cannot simulate image with num_frames < 0")

#     # Create list of Spots objects, one for each frame
#     real_spots = [Spots(params.num_spots) for i in range(params.num_frames)]
#     # Determine initial number of molecules per spot
#     if params.max_spot_molecules == 1:
#         # Fixed number of molecules (1 per spot)
#         n_mols = np.array([1] * params.num_spots)
#     else:
#         # Random number of molecules per spot (1 to max_spot_molecules)
#         n_mols = np.array(random.randint(1, params.max_spot_molecules, params.num_spots))
#     # Array to store fractional intensity loss due to bleaching within a frame
#     n_mols_fractional_intensity = np.zeros(n_mols.shape)

#     # Initialise spot positions and intensity for the first frame (frame 0)
#     # Random x positions within frame width
#     real_spots[0].positions[:, 0] = random.rand(params.num_spots) * params.frame_size[0]
#     # Random y positions within frame height
#     real_spots[0].positions[:, 1] = random.rand(params.num_spots) * params.frame_size[1]
#     # Set initial intensity based on Isingle (all spots start bright)
#     real_spots[0].spot_intensity[:] = params.I_single
#     # Set frame number for the first frame's Spots object
#     real_spots[0].frame = 1 # Frame index starts at 1

#     # Calculate diffusion step size (S) based on diffusion coeff, time, and pixel size
#     S = np.sqrt(2 * params.diffusion_coeff * params.frame_time) / params.pixel_size

#     # Simulate subsequent frames (from frame 1 onwards)
#     for frame in range(1, params.num_frames):
#         # Set current frame number for Spots object
#         real_spots[frame].frame = frame
#         # Calculate current intensity based on remaining molecules and fractional loss
#         real_spots[frame].spot_intensity[:] = params.I_single * (n_mols + n_mols_fractional_intensity)
#         # Copy trajectory numbers from previous frame (needed for build_trajectories)
#         real_spots[frame].traj_num = real_spots[frame - 1].traj_num[:]
#         # Simulate diffusion: new position = old position + random step (normal dist)
#         real_spots[frame].positions = random.normal(
#             real_spots[frame - 1].positions, S, (params.num_spots, 2)
#         )

#         # --- Simulate Photobleaching ---
#         # Reset fractional intensity loss for the *next* frame
#         n_mols_fractional_intensity[:] = 0
#         # Iterate through each spot
#         for i in range(params.num_spots):
#             # Check if the spot has any molecules left
#             if n_mols[i] > 0:
#                 # Check each molecule within the spot for bleaching
#                 molecules_to_bleach = 0
#                 for j in range(n_mols[i]):
#                     # Check if this molecule bleaches in this frame
#                     if random.rand() < params.p_bleach_per_frame:
#                         molecules_to_bleach += 1
#                         # Determine fractional time it survives into *next* frame
#                         frac = random.rand()
#                         # Add fractional intensity contribution for the *next* frame
#                         n_mols_fractional_intensity[i] += frac # Apply to correct spot index
#                 # Update number of molecules for the *next* frame
#                 n_mols[i] -= molecules_to_bleach
#         # --- End Photobleaching ---

#     # --- Generate Image Stack ---
#     # Create ImageData object to store the simulated frames
#     image = ImageData()
#     image.initialise(params.num_frames, params.frame_size)

#     # Create coordinate grid for intensity calculation
#     x_pos, y_pos = np.meshgrid(range(params.frame_size[0]), range(params.frame_size[1]))
#     # Iterate through each frame to generate the image data
#     for frame in range(params.num_frames):
#         # Initialise empty frame data (uint16 for pixel values)
#         frame_data = np.zeros([params.frame_size[1], params.frame_size[0]]).astype(np.uint16)

#         # Add intensity contribution from each spot
#         for spot in range(params.num_spots):
#             # Calculate Gaussian intensity profile for the spot
#             # Intensity is scaled by spot_intensity / normalisation factor
#             spot_intensity_value = real_spots[frame].spot_intensity[spot]
#             # Check if spot intensity is non-zero before calculating Gaussian
#             if spot_intensity_value > 0:
#                 spot_data = (
#                     (spot_intensity_value / (2 * np.pi * params.spot_width**2))
#                     * np.exp(
#                         -(
#                             (x_pos - real_spots[frame].positions[spot, 0]) ** 2
#                             + (y_pos - real_spots[frame].positions[spot, 1]) ** 2
#                         )
#                         / (2 * params.spot_width ** 2) # Use simulation spot_width param
#                     )
#                 ).astype(np.uint16) # Convert to uint16 before adding
#                 # Add spot's Gaussian profile to the frame data
#                 frame_data += spot_data
#                 # Update the spot's intensity field to reflect the *sum* of its Gaussian profile
#                 # This might not be the intended use of spot_intensity later?
#                 real_spots[frame].spot_intensity[spot]=np.sum(spot_data)
#             else:
#                 # If spot intensity is zero (e.g., fully bleached), set stored intensity to 0
#                  real_spots[frame].spot_intensity[spot] = 0


#         # --- Add Noise ---
#         # Add Poisson noise (shot noise) based on the signal intensity
#         frame_data = random.poisson(frame_data).astype(np.uint16) # Ensure uint16 after Poisson
#         # Add Gaussian background noise (read noise, background fluorescence)
#         bg_noise = random.normal(params.bg_mean, params.bg_std, [params.frame_size[1], params.frame_size[0]])
#         # Add background noise, ensuring pixel values remain non-negative
#         frame_data += np.where(bg_noise > 0, bg_noise.astype(np.uint16), 0)
#         # --- End Noise Addition ---

#         # Store the final frame data in the ImageData object
#         image[frame] = frame_data

#     # --- Build and Save Trajectories ---
#     # Build trajectories from the simulated spot data
#     real_trajs = trajectories.build_trajectories(real_spots, params)

#     # Write simulated image stack and trajectories to files
#     image.write(params.name + ".tif")
#     trajectories.write_trajectories(real_trajs, params.name + '_simulated.csv') # Use CSV extension

#     # Return the simulated image data and trajectories
#     return image, real_trajs