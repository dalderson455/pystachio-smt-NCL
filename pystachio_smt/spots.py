# #! /usr/bin/env python3
# # -*- coding: utf-8 -*-
# # vim:fenc=utf-8
# #
# # Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
# #
# # Distributed under terms of the MIT license.

""" SPOTS - Spot characterisation and manipulation module

Description:
    spots.py contains the Spots class for characterising and manipulating data
    associated with bright spots in the image datasets provided.

Contains:
    class Spots

Author:
    Edward Higgins

Version: 0.2.1
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
# Import local algorithms module
from . import algorithms
# Import KDTree for efficient neighbour searches
from scipy.spatial import KDTree
# Import collections for defaultdict and deque (used in merging)
import collections
# Import multiprocessing utilities
import multiprocessing
import contextlib

# --- Worker function for calculating intensity of a single spot ---
def _calculate_single_spot_intensity_worker(args):
    # Unpack arguments: position, pixel data, frame dimensions, parameters
    position, image_pixels, frame_size, params = args
    frame_width, frame_height = frame_size
    # Get subarray radius from parameters
    r = params.subarray_halfwidth

    # Round position to integer indices for array slicing
    x_int, y_int = round(position[0]), round(position[1])

    # Check if spot is too close to edge for subarray extraction
    if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
        # Return 0 if spot is near edge
        return 0.0

    # Define subarray boundaries
    y_start, y_end = y_int - r, y_int + r + 1
    x_start, x_end = x_int - r, x_int + r + 1
    # Extract the subarray (region around the spot)
    tmp = image_pixels[y_start:y_end, x_start:x_end]

    # Create a circular mask centred in the subarray
    spotmask = np.zeros(tmp.shape, dtype=np.uint8)
    # Draw filled circle (value 1) within the mask
    cv2.circle(spotmask,
               (r, r), # Centre coordinates (r, r) in the subarray
               params.inner_mask_radius, # Radius from params
               1,    # Mask value (integer 1)
               -1)   # Fill the circle

    # Calculate background intensity from pixels *outside* the mask
    bg_pixels = tmp[spotmask == 0]
    # Check if any background pixels exist
    if bg_pixels.size > 0:
        bgintensity = np.mean(bg_pixels)
    else:
        # Default background to 0 if mask covers entire subarray
        bgintensity = 0

    # Subtract background from the subarray
    tmp_bg_corr = tmp - bgintensity
    # Calculate spot intensity by summing pixels *inside* the mask
    intensity = np.sum(tmp_bg_corr[spotmask == 1])

    # Return the calculated background-corrected intensity
    return intensity


# --- Worker function for calculating width of a single spot ---
def _calculate_single_spot_width_worker(args):
    # Unpack arguments: position, pixel data, frame dimensions, parameters
    position, image_pixels, frame_size, params = args
    frame_width, frame_height = frame_size
    # Get subarray radius and fallback width from parameters
    r = params.subarray_halfwidth
    fallback_width = params.psf_width # Default if fit fails

    # Round position to integer indices
    x_int, y_int = round(position[0]), round(position[1])

    # Check if spot is too close to edge
    if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
        # Return fallback width and failure flag if near edge
        return ([fallback_width, fallback_width], 0)

    # Define subarray boundaries
    y_start, y_end = y_int - r, y_int + r + 1
    x_start, x_end = x_int - r, x_int + r + 1
    # Extract subarray
    tmp = image_pixels[y_start:y_end, x_start:x_end]

    # Create circular mask (needed for background calculation)
    spotmask = np.zeros(tmp.shape, dtype=np.uint8)
    cv2.circle(spotmask, (r, r), params.inner_mask_radius, 1, -1)

    # Calculate background intensity from outside the mask
    bg_pixels = tmp[spotmask == 0]
    if bg_pixels.size > 0:
        bgintensity = np.mean(bg_pixels)
    else:
        bgintensity = 0

    # Subtract background from subarray
    tmp_bg_corr = tmp - bgintensity

    # Fit 2D Gaussian to the background-corrected subarray
    p, succ = algorithms.fit2Dgaussian(tmp_bg_corr)

    # Check if fit succeeded and widths (p[3], p[4]) are positive
    if succ == 1 and p[3] > 0 and p[4] > 0:
        # Store fitted widths [wx, wy]
        width_xy = [p[3], p[4]]
        success_flag = 1 # Indicate success
    else:
        # Use fallback widths if fit failed or widths non-physical
        width_xy = [fallback_width, fallback_width]
        success_flag = 0 # Indicate failure

    # Return calculated/fallback widths and success flag
    return (width_xy, success_flag)

# --- Worker function for refining centre of a single spot ---
def _refine_single_spot_worker(args):
    # Unpack arguments: initial position, pixel data, frame dimensions, parameters
    initial_position, image_pixels, frame_size, params = args

    # Work on a copy of the initial position
    p_estimate = initial_position.copy()
    frame_width, frame_height = frame_size
    # Get subarray radius from parameters
    r = params.subarray_halfwidth
    N = 2 * r + 1 # Subarray dimension

    # --- Initial Checks ---
    spot_failed = False
    # Check if initial position is too close to any edge
    for d in (0, 1): # Check x (0) and y (1)
        if round(p_estimate[d]) < r or round(p_estimate[d]) >= frame_size[d] - r:
            # print(f"Warning: Initial position {initial_position} too close to edge for refinement.")
            spot_failed = True
            break # Exit check loop
    # Return default values if initial position is invalid
    if spot_failed:
        return (initial_position, 0.0, 0.0, 0.0, False)

    # --- Iterative Refinement Loop ---
    converged = False
    iteration = 0
    # Initialise return values (in case of early exit)
    bg_average = 0.0
    spot_intensity = 0.0
    snr = 0.0

    # Loop until converged or max iterations reached
    while not converged and iteration < params.gauss_mask_max_iter:
        iteration += 1

        # Get current integer coordinates based on p_estimate
        current_x_int, current_y_int = round(p_estimate[0]), round(p_estimate[1])

        # Check if current estimate is too close to edge *before* slicing
        if not (r <= current_x_int < frame_width - r and r <= current_y_int < frame_height - r):
            #  print(f"Warning: Spot drifted too close to edge during refinement ({p_estimate}). Using last valid estimate.")
             # Stop iteration, result is not reliably converged
             converged = False
             break

        # Define subarray slice indices
        y_start, y_end = current_y_int - r, current_y_int + r + 1
        x_start, x_end = current_x_int - r, current_x_int + r + 1

        # Extract current subarray pixels
        spot_pixels = image_pixels[y_start:y_end, x_start:x_end]

        # Create coordinate grids relative to *full image* for calculations
        Xs, Ys = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

        # --- Mask Generation ---
        # Use full image coordinates for mask calculations relative to p_estimate
        coords_for_mask_x, coords_for_mask_y = Xs, Ys

        # Create circular inner mask (pixels within inner_mask_radius)
        inner_mask = np.where(
            (coords_for_mask_x - p_estimate[0])**2 + (coords_for_mask_y - p_estimate[1])**2 <= params.inner_mask_radius**2,
            1, 0
        ).astype(np.uint8) # Use uint8 for mask type

        # Create Gaussian weighting mask centred on p_estimate
        exponent = -((coords_for_mask_x - p_estimate[0])**2 + (coords_for_mask_y - p_estimate[1])**2) / (2 * params.gauss_mask_sigma**2)
        gauss_mask = np.exp(exponent)

        # Normalise Gaussian mask (avoid division by zero)
        gauss_sum = np.sum(gauss_mask)
        if gauss_sum != 0:
            gauss_mask /= gauss_sum
        else:
            # If sum is zero, set mask to zero to prevent NaN later
            gauss_mask = np.zeros_like(gauss_mask)

        # Create background mask (inverse of inner mask)
        bg_mask = 1 - inner_mask

        # --- Calculations ---
        # Calculate local background mean and standard deviation
        spot_bg_pixels = spot_pixels[bg_mask == 1]
        if spot_bg_pixels.size > 0:
             bg_average = np.mean(spot_bg_pixels)
             bg_std = np.std(spot_bg_pixels)
        else:
             # Defaults if no background pixels found
             bg_average = 0.0
             bg_std = 0.0

        # Create background-corrected subarray
        bg_corr_spot_pixels = spot_pixels - bg_average

        # Calculate new position estimate using intensity-weighted centroid with Gaussian mask
        spot_gaussian_product = bg_corr_spot_pixels * gauss_mask
        sum_spot_gaussian_product = np.sum(spot_gaussian_product)

        # Default to old estimate if weighted sum is zero
        p_estimate_new = p_estimate.copy()
        if sum_spot_gaussian_product != 0:
            # Calculate weighted average x and y coordinates
            p_estimate_new[0] = np.sum(spot_gaussian_product * Xs) / sum_spot_gaussian_product
            p_estimate_new[1] = np.sum(spot_gaussian_product * Ys) / sum_spot_gaussian_product
        else:
             # Handle division by zero case
             print(f"Warning: Sum of weighted pixels is zero for spot refinement ({p_estimate}).")
             converged = False # Cannot converge
             break # Exit loop

        # Check for NaN in new estimate (should be less likely now)
        if np.isnan(p_estimate_new).any():
            print(f"Warning: Position estimate became NaN for spot ({p_estimate}), using previous value.")
            converged = False # Cannot converge
            break # Exit loop

        # Calculate change in position estimate from previous iteration
        estimate_change = np.linalg.norm(p_estimate - p_estimate_new)
        # Update position estimate for next iteration or final result
        p_estimate = p_estimate_new

        # Calculate current spot intensity and SNR
        # Intensity is sum of background-corrected pixels within inner mask
        spot_intensity = np.sum(bg_corr_spot_pixels[inner_mask == 1])

        # Calculate SNR (avoid division by zero)
        denominator_snr = bg_std * np.sum(inner_mask)
        if denominator_snr != 0:
             snr = abs(spot_intensity / denominator_snr)
        else:
             # Default SNR if background std dev or mask area is zero
             snr = 0.0

        # Check for convergence based on position change threshold
        if estimate_change < 1e-6:
            converged = True

        # Optional: Exit early if SNR drops too low before converging
        if snr <= params.snr_filter_cutoff and not converged:
            break # Exit loop

    # --- End of Refinement Loop ---

    # Return final position, background, intensity, SNR, and convergence status
    return (p_estimate, bg_average, spot_intensity, snr, converged)

# Class definition for storing and manipulating spot data within a frame
class Spots:
    # Initialiser for the Spots object
    def __init__(self, num_spots=0, frame=0):
        # Store number of spots and frame number
        self.num_spots = num_spots
        self.frame = frame
        # If num_spots > 0, initialise arrays for spot attributes
        if num_spots > 0:
            self.positions = np.zeros([num_spots, 2]) # [x, y] coordinates
            self.bg_intensity = np.zeros(num_spots) # Background intensity
            self.spot_intensity = np.zeros(num_spots) # Spot intensity
            self.traj_num = [-1] * self.num_spots # Trajectory ID (-1 initially)
            self.snr = np.zeros(num_spots) # Signal-to-noise ratio
            self.laser_on_frame = 0 # Placeholder (seems unused here)
            self.converged = np.zeros(num_spots, dtype=np.int8) # Refinement convergence flag
            self.exists = True # Flag indicating object contains data
            self.width = np.zeros((num_spots,2)) # Spot width [width_x, width_y]
            self.centre_intensity = np.zeros(num_spots) # Intensity at centre pixel (unused?)
        else:
            # If num_spots is 0, set exists flag to False
            self.exists = False

    # Method to set/update spot positions and reset related attributes
    def set_positions(self, positions):
        # Update number of spots based on input positions array
        self.num_spots = len(positions)
        # Re-initialise all attribute arrays based on the new number of spots
        self.positions = np.zeros([self.num_spots, 2])
        self.clipping = [False] * self.num_spots # Clipping flag (unused?)
        self.bg_intensity = np.zeros(self.num_spots)
        self.spot_intensity = np.zeros(self.num_spots)
        self.width = np.zeros([self.num_spots, 2])
        self.traj_num = [-1] * self.num_spots
        self.snr = np.zeros(self.num_spots)
        self.converged = np.zeros(self.num_spots,dtype=np.int8)
        self.centre_intensity = np.zeros(self.num_spots) # (unused?)

        # Populate the positions array with the input data
        for i in range(self.num_spots):
            self.positions[i, :] = positions[i]

    # Method to calculate background-corrected spot intensities
    def get_spot_intensities(self, frame, params):
        # Check if there are any spots to process
        if self.num_spots == 0:
            self.spot_intensity = np.array([]) # Ensure attribute exists
            return

        # Get pixel data and frame size, handling both ImageData and ndarray inputs
        if hasattr(frame, 'as_image') and callable(frame.as_image):
            image_pixels = frame.as_image()
            frame_size = frame.frame_size
        elif isinstance(frame, np.ndarray):
            image_pixels = frame
            frame_size = (frame.shape[1], frame.shape[0]) # Assume (height, width) -> (width, height)
        else:
            print("Error: Unsupported frame data type in get_spot_intensities.")
            self.spot_intensity = np.zeros(self.num_spots) # Assign default
            return

        # Determine if multiprocessing should be used internally
        # Only run in parallel if requested AND not already in a worker process
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Prepare arguments for each spot calculation task
        task_args = []
        for i in range(self.num_spots):
            task_args.append((self.positions[i], image_pixels, frame_size, params))

        # Execute tasks either in parallel or serially
        if run_parallel_internally:
            # Use multiprocessing pool
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                results = pool.starmap(_calculate_single_spot_intensity_worker, task_args)
        else:
            # Run serially
            results = []
            for args in task_args:
                results.append(_calculate_single_spot_intensity_worker(args))

        # Store the calculated intensities in the object's attribute
        self.spot_intensity = np.array(results)

    # Method to calculate spot widths using 2D Gaussian fitting
    def get_spot_widths(self, frame, params):
        # Check if there are spots to process
        if self.num_spots == 0:
            self.width = np.array([]) # Ensure attribute exists
            return

        # Get pixel data and frame size
        if hasattr(frame, 'as_image') and callable(frame.as_image):
            image_pixels = frame.as_image()
            frame_size = frame.frame_size
        elif isinstance(frame, np.ndarray):
            image_pixels = frame
            frame_size = (frame.shape[1], frame.shape[0]) # Assume (height, width) -> (width, height)
        else:
            print("Error: Unsupported frame data type in get_spot_widths.")
            # Assign default widths based on psf_width parameter
            self.width = np.full((self.num_spots, 2), params.psf_width)
            return

        # Determine if multiprocessing should be used
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Prepare arguments for each spot width calculation
        task_args = []
        for i in range(self.num_spots):
            task_args.append((self.positions[i], image_pixels, frame_size, params))

        # Execute tasks in parallel or serially
        if run_parallel_internally:
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                results = pool.starmap(_calculate_single_spot_width_worker, task_args)
        else:
            results = []
            for args in task_args:
                results.append(_calculate_single_spot_width_worker(args))

        # --- Process results and update attributes ---
        new_widths = np.zeros((self.num_spots, 2))
        success_flags = np.zeros(self.num_spots, dtype=int)

        # Unpack results (width_xy, success_flag) for each spot
        for i, result in enumerate(results):
            width_xy, success_flag = result
            new_widths[i, :] = width_xy
            success_flags[i] = success_flag

        # Store the calculated/fallback widths
        self.width = new_widths

        # Calculate and return the number of successful fits and total processed
        num_success = np.sum(success_flags)
        total_processed = self.num_spots
        return num_success, total_processed

    # --- Helper method for image preparation ---
    def _prepare_image(self, frame: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
        # Convert grayscale frame to BGR (required by some OpenCV functions)
        img_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Create elliptical structuring element (disk) for morphological operations
        disk_size = 2 * params.struct_disk_radius - 1 # Diameter
        disk_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (disk_size, disk_size)
        )

        # Apply Gaussian blur if specified in parameters
        if params.filter_image == "Gaussian":
            prepared_frame = cv2.GaussianBlur(img_frame_bgr, (3, 3), 0) # 3x3 kernel
        else:
            # Otherwise, use the original BGR frame
            prepared_frame = img_frame_bgr.copy()

        # Return the prepared frame and the structuring element
        return prepared_frame, disk_kernel

    # --- Helper method for threshold calculation ---
    def _calculate_threshold(self, image: np.ndarray, disk_kernel, params) -> tuple[int, np.ndarray]:
        # Apply morphological Top-Hat filter to enhance bright spots on dark background
        tophatted_frame = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, disk_kernel)

        # Calculate histogram of the top-hat filtered image (channel 0)
        hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])
        # Ignore the first bin (often background noise)
        hist_data[0] = 0

        # Calculate Full Width at Half Maximum (FWHM) and peak location of histogram
        peak_width, peak_location = algorithms.fwhm(hist_data)
        # Determine binary threshold based on peak location and width tolerance
        bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)

        # Return the calculated threshold and the top-hat image
        return bw_threshold, tophatted_frame

    # --- Helper method for creating binary mask ---
    def _create_binary_mask(self, tophatted_frame: np.ndarray, threshold: int, params) -> np.ndarray:
        # Apply Gaussian blur to the top-hat image to smooth noise
        blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)

        # Apply binary thresholding using the calculated threshold
        bw_frame = cv2.threshold(
            blurred_tophatted_frame, threshold, 255, cv2.THRESH_BINARY
        )[1] # Get the thresholded image (second element of tuple)

        # Apply morphological Opening (erosion then dilation) to remove small noise pixels
        bw_opened = cv2.morphologyEx(
            bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # Use 3x3 cross kernel
        )

        # Apply morphological Closing (dilation then erosion) to fill small holes in spots
        bw_filled = cv2.morphologyEx(
            bw_opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), # Use 3x3 elliptical kernel
        )
        # Return the first channel of the result (it's binary/grayscale)
        return bw_filled[:, :, 0]

    # --- Helper method for detecting spot centres ---
    def _detect_spot_centres(self, binary_mask: np.ndarray) -> np.ndarray:
        # Use ultimate erosion algorithm (find peaks in distance transform)
        spot_locations = algorithms.ultimate_erode_scipy(binary_mask)
        # Check for NaN values in the result (shouldn't happen with scipy version)
        if np.isnan(spot_locations).any():
            print("Warning: NaNs detected after ultimate erosion.")
            # Return empty array if NaNs found
            return np.array([])
        # Return array of [x, y] coordinates
        return spot_locations

    # --- Main public method for finding spots in a frame ---
    def find_in_frame(self, frame: np.ndarray, params):
        # 1. Image Preparation: Convert to BGR, apply blur, create kernel
        prepared_frame, disk_kernel = self._prepare_image(frame, params)

        # 2. Threshold Calculation: Apply top-hat, analyse histogram
        threshold, tophatted_frame = self._calculate_threshold(prepared_frame, disk_kernel, params)

        # 3. Binary Mask Creation: Blur, threshold, open, close
        binary_mask = self._create_binary_mask(tophatted_frame, threshold, params)

        # 4. Spot Centre Detection: Use ultimate erosion on the binary mask
        spot_locations = self._detect_spot_centres(binary_mask)

        # 5. Store Results: Update the object's positions attribute
        self.set_positions(spot_locations)

    # Method to merge spots that are very close together
    def merge_coincident_candidates(self):
        # Skip if fewer than 2 spots exist
        if self.num_spots < 2:
            return

        # Define merge distance threshold (sqrt(2) pixels)
        merge_radius = np.sqrt(2.0)

        # Build KDTree for efficient nearest neighbour search
        tree = KDTree(self.positions)

        # Find all pairs of spots closer than merge_radius
        close_pairs = tree.query_pairs(r=merge_radius)

        # Skip if no pairs are close enough
        if not close_pairs:
            return

        # --- Group close pairs into clusters using graph connectivity (connected components) ---
        # Build adjacency list representation of the graph where nodes are spot indices
        adj = collections.defaultdict(list)
        for i, j in close_pairs:
            adj[i].append(j)
            adj[j].append(i)

        # Find connected components (clusters) using Breadth-First Search (BFS)
        clusters = [] # List to store clusters (each cluster is a list of spot indices)
        visited = set() # Keep track of visited spot indices

        # Iterate through all spot indices
        for i in range(self.num_spots):
            # If spot 'i' is part of a close pair and not yet visited, start BFS
            if i not in visited and i in adj:
                current_cluster = set() # Store indices in the current cluster
                q = collections.deque([i]) # Queue for BFS
                visited.add(i)
                current_cluster.add(i)

                # Perform BFS
                while q:
                    u = q.popleft() # Get current node
                    # Explore neighbours
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            current_cluster.add(v)
                            q.append(v)
                # Add the found cluster to the list
                clusters.append(list(current_cluster))
            # If spot 'i' was not part of any close pair and not visited, add as a single-spot cluster
            elif i not in visited and i not in adj:
                clusters.append([i])
                visited.add(i) # Mark as visited
        # --- End of cluster finding ---

        # --- Calculate new positions based on cluster centroids ---
        new_positions = []
        # Iterate through the found clusters
        for cluster_indices in clusters:
            # If cluster has only one spot, keep original position
            if len(cluster_indices) == 1:
                spot_index = cluster_indices[0]
                new_positions.append(self.positions[spot_index])
            # If cluster has multiple spots, calculate centroid
            else:
                cluster_pos = self.positions[cluster_indices] # Get positions of spots in cluster
                mean_pos = np.mean(cluster_pos, axis=0) # Calculate mean position (centroid)
                new_positions.append(mean_pos)

        # Update the object's positions with the new (potentially merged) positions
        if not new_positions:
             # Handle unlikely case where merging leaves no spots
            print("Warning: Merging resulted in no spots left.")
            self.set_positions(np.array([]))
        else:
            self.set_positions(np.array(new_positions))

    # Method to filter spots based on various criteria
    def filter_candidates(self, frame, params):
        # Skip if no spots exist
        if self.num_spots == 0:
            return

        # --- Create Boolean Mask for Filtering ---
        # Start assuming all spots will be kept
        keep_mask = np.ones(self.num_spots, dtype=bool)

        # 1. Filter by Signal-to-Noise Ratio (SNR)
        # Keep only spots with SNR above the cutoff
        keep_mask &= (self.snr > params.snr_filter_cutoff)

        # 2. Filter by Proximity to Image Edges
        # Get frame dimensions and subarray halfwidth
        frame_height, frame_width = frame.frame_size[1], frame.frame_size[0]
        hw = params.subarray_halfwidth
        # Check if x-coordinate is too close to left or right edge
        too_close_x = (self.positions[:, 0] < hw) | (self.positions[:, 0] >= frame_width - hw)
        # Check if y-coordinate is too close to top or bottom edge
        too_close_y = (self.positions[:, 1] < hw) | (self.positions[:, 1] >= frame_height - hw)
        # Keep spots that are *not* too close to any edge
        keep_mask &= ~(too_close_x | too_close_y)

        # 3. Filter by External Mask (if provided)
        if frame.has_mask:
            # Round spot coordinates to nearest integer for mask indexing
            # Clip coordinates to stay within mask boundaries
            rounded_y = np.clip(np.round(self.positions[:, 1]).astype(int), 0, frame_height - 1)
            rounded_x = np.clip(np.round(self.positions[:, 0]).astype(int), 0, frame_width - 1)

            # Try to get mask values at spot locations
            try:
                # Keep spots where the mask value is non-zero
                mask_values = frame.mask_data[rounded_y, rounded_x]
                keep_mask &= (mask_values != 0)
            # Handle potential errors if coordinates are out of bounds (shouldn't happen with clip)
            except IndexError:
                print("Warning: Index error during mask filtering.")
                # Decide how to handle - here we just skip mask filtering if error occurs
                pass

        # --- Apply the Combined Boolean Mask ---
        # Filter all relevant spot attributes using the final keep_mask
        self.positions = self.positions[keep_mask]
        self.snr = self.snr[keep_mask]
        # Check and filter other attributes if they exist and are numpy arrays
        if hasattr(self, 'clipping') and isinstance(self.clipping, np.ndarray):
            self.clipping = self.clipping[keep_mask]
        if hasattr(self, 'bg_intensity') and isinstance(self.bg_intensity, np.ndarray):
            self.bg_intensity = self.bg_intensity[keep_mask]
        if hasattr(self, 'spot_intensity') and isinstance(self.spot_intensity, np.ndarray):
            self.spot_intensity = self.spot_intensity[keep_mask]
        if hasattr(self, 'centre_intensity') and isinstance(self.centre_intensity, np.ndarray):
            self.centre_intensity = self.centre_intensity[keep_mask]
        if hasattr(self, 'width') and isinstance(self.width, np.ndarray):
            self.width = self.width[keep_mask]
        # Handle traj_num (might be list or array)
        if hasattr(self, 'traj_num') and isinstance(self.traj_num, np.ndarray):
            self.traj_num = self.traj_num[keep_mask]
        elif hasattr(self, 'traj_num') and isinstance(self.traj_num, list):
            # Filter list using list comprehension if it's not an array
             self.traj_num = [self.traj_num[i] for i, keep in enumerate(keep_mask) if keep]
        if hasattr(self, 'converged') and isinstance(self.converged, np.ndarray):
            self.converged = self.converged[keep_mask]

        # Update the total number of spots after filtering
        self.num_spots = self.positions.shape[0]

    # Method to calculate pairwise distances between spots in this object and another Spots object
    def distance_from(self, other):
        # Initialise matrix to store distances [num_spots_self, num_spots_other]
        distances = np.zeros([self.num_spots, other.num_spots])

        # Calculate Euclidean distance for each pair of spots
        for i in range(self.num_spots):
            for j in range(other.num_spots):
                # Distance = sqrt( (x1-x2)^2 + (y1-y2)^2 )
                distances[i, j] = np.linalg.norm(
                    self.positions[i, :] - other.positions[j, :]
                )
        # Return the distance matrix
        return distances

    # Method to assign initial trajectory numbers (0 to num_spots-1) - typically used for the first frame
    def index_first(self):
        self.traj_num = list(range(self.num_spots))

    # Method to link spots in the current frame to spots in the previous frame
    def link(self, prev_spots, params):
        # Calculate distances between all current spots and all previous spots
        distances = self.distance_from(prev_spots)

        # Find the indices of previous spots sorted by distance for each current spot
        neighbours = np.argsort(distances[:, :], axis=1)
        # Keep track of previous spots that have already been linked
        paired_spots = []
        # Determine the next available trajectory ID
        next_trajectory = max(prev_spots.traj_num) + 1 if prev_spots.num_spots > 0 else 0 # Handle empty prev_spots

        # Iterate through each spot in the current frame
        for i in range(self.num_spots):
            linked = False # Flag to check if spot 'i' gets linked
            # Iterate through potential matches (nearest previous spots first)
            for j in range(prev_spots.num_spots):
                neighbour_idx = neighbours[i, j] # Index of the j-th nearest previous spot
                # Check if the distance to this neighbour is within the maximum allowed displacement
                if distances[i, neighbour_idx] < params.max_displacement:
                    # Check if this neighbour has already been paired with another current spot
                    if neighbour_idx in paired_spots:
                        # If already paired, continue to the next nearest neighbour
                        continue
                    else:
                        # If not paired, link spot 'i' to this neighbour
                        paired_spots.append(neighbour_idx) # Mark neighbour as paired
                        self.traj_num[i] = prev_spots.traj_num[neighbour_idx] # Assign same trajectory ID
                        linked = True # Mark spot 'i' as linked
                        break # Move to the next current spot 'i'
                else:
                    # If the nearest unpaired neighbour is too far, assign a new trajectory ID
                    self.traj_num[i] = next_trajectory
                    next_trajectory += 1 # Increment ID for the next new trajectory
                    linked = True # Mark spot 'i' as linked (to a new trajectory)
                    break # Move to the next current spot 'i'

            # If after checking all neighbours, the spot wasn't linked (should only happen if prev_spots is empty)
            if not linked and prev_spots.num_spots == 0:
                 self.traj_num[i] = next_trajectory
                 next_trajectory += 1
                 linked = True


            # Error check: Ensure every spot gets assigned a trajectory ID (should always happen)
            if self.traj_num[i] == -1:
                sys.exit(f"ERROR: Unable to find a match or assign new trajectory for spot {i}, frame {self.frame}")

    # Method to refine spot centres using iterative centroid calculation
    def refine_centres(self, frame, params):
        # Check if there are spots to refine
        if self.num_spots == 0:
            return

        # Get image pixel data and frame size once
        image_pixels = frame.as_image()
        frame_size = frame.frame_size
        # Determine if multiprocessing should be used
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Prepare arguments for each spot refinement task
        task_args = []
        for i in range(self.num_spots):
            task_args.append((self.positions[i], image_pixels, frame_size, params))

        # Execute tasks in parallel or serially
        if run_parallel_internally:
            # Use multiprocessing pool
            # print(f"Running refine_centres internally in parallel with {params.num_procs} processes.") # Optional log
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                results = pool.starmap(_refine_single_spot_worker, task_args)
        else:
            # Optional log for serial execution
            # if multiprocessing.current_process().daemon: print("Running refine_centres internally in serial mode (already in worker process).")
            # else: print("Running refine_centres internally in serial mode.")
            # Run serially
            results = []
            for args in task_args:
                results.append(_refine_single_spot_worker(args))

        # --- Unpack results and update object attributes ---
        # Initialise arrays to store results
        new_positions = np.zeros_like(self.positions)
        new_bg_intensity = np.zeros_like(self.bg_intensity)
        new_spot_intensity = np.zeros_like(self.spot_intensity)
        new_snr = np.zeros_like(self.snr)
        new_converged = np.zeros_like(self.converged, dtype=bool) # Use bool initially

        # Iterate through results and populate arrays
        for i, result in enumerate(results):
            final_pos, bg_avg, spot_int, snr_val, conv_status = result
            new_positions[i, :] = final_pos
            new_bg_intensity[i] = bg_avg
            new_spot_intensity[i] = spot_int
            new_snr[i] = snr_val
            new_converged[i] = conv_status

        # Update object attributes with the refined values
        self.positions = new_positions
        self.bg_intensity = new_bg_intensity
        self.spot_intensity = new_spot_intensity
        self.snr = new_snr
        # Convert convergence status back to int8 if required by downstream code
        self.converged = new_converged.astype(np.int8)