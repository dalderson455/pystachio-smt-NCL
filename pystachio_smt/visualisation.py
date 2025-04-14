# #! /usr/bin/env python3
# # -*- coding: utf-8 -*-
# # vim:fenc=utf-8
# #
# # Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
# #
# # Distributed under terms of the MIT license.

""" VISUALISATION - Module for creating visual outputs

Description:
    visualisation.py contains code for rendering image data and trajectories,
    primarily for creating animations or static plots showing spot movement.

Contains:
    function render

Author:
    Edward Higgins

Version: 0.2.1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
# Import local modules
from . import images
from . import trajectories

# Function to render an animation of image frames with overlaid trajectories
def render(params):
    # Create ImageData object and read the image file
    image_data = images.ImageData()
    image_data.read(params.name+".tif", params) # Assumes TIF extension

    # Attempt to read tracked trajectories (handling potential errors)
    try:
        trajs = trajectories.read_trajectories(params.name + "_trajectories.csv") # Use CSV
        if trajs is None: trajs = [] # Use empty list if file read fails
    except Exception as e:
        print(f"Warning: Could not read tracked trajectories: {e}")
        trajs = []

    # Attempt to read simulated (ground truth) trajectories (handling potential errors)
    try:
        true_trajs = trajectories.read_trajectories(params.name + "_simulated.csv") # Use CSV
        if true_trajs is None: true_trajs = [] # Use empty list if file read fails
    except Exception as e:
        print(f"Warning: Could not read simulated trajectories: {e}")
        true_trajs = []

    # Find maximum intensity value across all frames for consistent colour scaling
    maxval = image_data.max_intensity()

    # Create Matplotlib figure for the animation/plot
    figure = plt.figure()

    # Get pixel size and frame dimensions from parameters/image data
    px_size = params.pixel_size
    fr_size = image_data.frame_size # Assuming [width, height]

    # Get a list of distinct colours for plotting trajectories
    colors = list(mcolors.TABLEAU_COLORS.keys())

    # --- Plot Tracked Trajectories (if available) ---
    if trajs:
        # Print message if verbose mode is on
        if params.verbose:
            print(f"Displaying {len(trajs)} tracked trajectories")
        # Iterate through each tracked trajectory
        for traj in trajs:
            # Extract x and y coordinates from the trajectory path
            x = []
            y = []
            # Loop through frames the trajectory exists in
            # Note: range excludes end_frame, use end_frame + 1 to include it
            for frame_num in range(traj.start_frame, traj.end_frame + 1):
                idx = frame_num - traj.start_frame # Calculate index
                if idx < len(traj.path): # Check index validity
                    x.append(traj.path[idx][0])
                    y.append(traj.path[idx][1])
                else:
                    print(f"Warning: Index mismatch plotting tracked traj {traj.id} at frame {frame_num}")

            # Convert coordinates to numpy arrays
            x = np.array(x)
            y = np.array(y)
            # Scale coordinates from pixels to physical units (e.g., μm)
            # Add 0.5 to centre coordinate in pixel before scaling
            # Invert y-axis for typical image display (origin top-left)
            x_scaled = (x + 0.5) * px_size
            y_scaled = (fr_size[1] - y - 0.5) * px_size
            # Plot the trajectory path with circles and lines, cycling through colours
            plt.plot(x_scaled, y_scaled, "o-", c=colors[traj.id % len(colors)], markersize=3, linewidth=1) # Smaller markers/lines

    # --- Plot Simulated/True Trajectories (if available) ---
    if true_trajs:
        # Print message if verbose
        if params.verbose:
            print(f"Displaying {len(true_trajs)} true trajectories")
        # Iterate through each true trajectory
        for traj in true_trajs:
            # Extract x and y coordinates
            x = []
            y = []
            # Loop through frames the trajectory exists in
            for frame_num in range(traj.start_frame, traj.end_frame + 1):
                idx = frame_num - traj.start_frame # Calculate index
                if idx < len(traj.path): # Check index validity
                    x.append(traj.path[idx][0])
                    y.append(traj.path[idx][1])
                else:
                     print(f"Warning: Index mismatch plotting true traj {traj.id} at frame {frame_num}")

            # Convert coordinates to numpy arrays
            x = np.array(x)
            y = np.array(y)
            # Scale coordinates to physical units and invert y-axis
            x_scaled = (x + 0.5) * px_size
            y_scaled = (fr_size[1] - y - 0.5) * px_size
            # Plot the true trajectory path with crosses and dashed lines
            plt.plot(x_scaled, y_scaled, "+--", c="tab:orange", markersize=4, linewidth=1) # Distinct style

    # --- Create Animation Frames ---
    plts = [] # List to hold plot objects for each animation frame
    # Iterate through each frame in the image data
    for frame in range(image_data.num_frames):
        # Display the image frame using imshow
        plt_frame = plt.imshow(
                image_data.pixel_data[frame,:,:], # Pixel data for the current frame
                vmin=0,                           # Minimum intensity for colour scale
                animated=True,                    # Required for animation
                vmax=maxval,                      # Maximum intensity (ensures consistent scaling)
                cmap=plt.get_cmap("gray"),        # Grayscale colour map
                # Set extent to match physical units (microns)
                extent=[0, fr_size[0]*px_size, 0, fr_size[1]*px_size]
        )
        # Add the plot object for this frame to the list (needs to be in a list itself)
        plts.append([plt_frame])

    # Create the animation object
    ani = animation.ArtistAnimation(
            figure,       # The Matplotlib figure
            plts,         # List of frame plot objects
            interval=50, # Delay between frames in milliseconds
            blit=True     # Use blitting for performance (optional)
            )

    # --- Final Plot Formatting and Display ---
    # Set title (adjust frame number display if needed)
    plt.title(f"Tracking Visualisation (Frame 1 to {image_data.num_frames})")
    # Label axes with units
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    # Add a colour bar indicating intensity scale
    plt.colorbar(label='Intensity (counts)') # Add label to colorbar
    # Adjust aspect ratio to be equal (so pixels look square)
    plt.gca().set_aspect('equal', adjustable='box')
    # Show the plot window with the animation
    plt.show()