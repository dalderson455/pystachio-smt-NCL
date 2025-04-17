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

# --- Core library imports ---
import sys
import cv2
import numpy as np
import collections
import multiprocessing
import contextlib

# --- SciPy and Numba imports ---
from scipy.spatial import KDTree
from numba import jit

# --- Local module imports ---
from . import algorithms


# --- Worker func: Calculate single spot intensity ---
def _calculate_single_spot_intensity_worker(args):
    # Unpack inputs
    position, image_pixels, frame_size, params = args
    frame_width, frame_height = frame_size
    r = params.subarray_halfwidth

    # Get integer coords & check bounds
    x_int, y_int = round(position[0]), round(position[1])
    if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
        return 0.0 # Near edge

    # Extract subarray
    y_start, y_end = y_int - r, y_int + r + 1
    x_start, x_end = x_int - r, x_int + r + 1
    tmp = image_pixels[y_start:y_end, x_start:x_end]

    # Create circular mask
    spotmask = np.zeros(tmp.shape, dtype=np.uint8)
    cv2.circle(spotmask, (r, r), params.inner_mask_radius, 1, -1)

    # Calculate background from annulus
    bg_pixels = tmp[spotmask == 0]
    bgintensity = np.mean(bg_pixels) if bg_pixels.size > 0 else 0

    # Subtract background and sum intensity within mask
    tmp_bg_corr = tmp - bgintensity
    intensity = np.sum(tmp_bg_corr[spotmask == 1])

    return intensity


# --- Worker func: Calculate single spot width via Gaussian fit ---
def _calculate_single_spot_width_worker(args):
    # Unpack inputs
    position, image_pixels, frame_size, params = args
    frame_width, frame_height = frame_size
    r = params.subarray_halfwidth
    fallback_width = params.psf_width

    # Get integer coords & check bounds
    x_int, y_int = round(position[0]), round(position[1])
    if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
        return ([fallback_width, fallback_width], 0) # Near edge

    # Extract subarray
    y_start, y_end = y_int - r, y_int + r + 1
    x_start, x_end = x_int - r, x_int + r + 1
    tmp = image_pixels[y_start:y_end, x_start:x_end]

    # Create mask for background calc
    spotmask = np.zeros(tmp.shape, dtype=np.uint8)
    cv2.circle(spotmask, (r, r), params.inner_mask_radius, 1, -1)

    # Calculate background from annulus
    bg_pixels = tmp[spotmask == 0]
    bgintensity = np.mean(bg_pixels) if bg_pixels.size > 0 else 0

    # Subtract background
    tmp_bg_corr = tmp - bgintensity

    # Fit 2D Gaussian
    p, succ = algorithms.fit2Dgaussian(tmp_bg_corr)

    # Check fit success and assign width
    if succ == 1 and p[3] > 0 and p[4] > 0:
        width_xy = [p[3], p[4]]
        success_flag = 1
    else:
        width_xy = [fallback_width, fallback_width]
        success_flag = 0

    return (width_xy, success_flag)

# --- Worker func: Refine single spot centre (Numba optimised) ---
@jit(nopython=True)
def _refine_single_spot_worker(args):
    # Unpack inputs (scalars/arrays for Numba)
    initial_position, image_pixels, frame_size_tuple, \
    subarray_halfwidth, inner_mask_radius, gauss_mask_sigma, \
    gauss_mask_max_iter, snr_filter_cutoff = args

    p_estimate = initial_position.copy()
    frame_width, frame_height = frame_size_tuple
    r = subarray_halfwidth

    # Initial boundary check
    spot_failed = False
    for d in (0, 1):
        p_estimate_d_int = int(p_estimate[d] + 0.5)
        if p_estimate_d_int < r or p_estimate_d_int >= frame_size_tuple[d] - r:
            spot_failed = True
            break
    if spot_failed:
        return (initial_position, 0.0, 0.0, 0.0, False)

    # Initialise loop variables
    converged = False
    iteration = 0
    bg_average = 0.0
    spot_intensity = 0.0
    snr = 0.0
    p_estimate_prev = p_estimate.copy()

    # Iterative refinement loop
    while not converged and iteration < gauss_mask_max_iter:
        iteration += 1

        # Check for oscillation or NaN
        if iteration > 5 and np.linalg.norm(p_estimate - p_estimate_prev) < 1e-9 and not converged: break
        if np.any(np.isnan(p_estimate)):
            p_estimate = p_estimate_prev
            converged = False
            break
        p_estimate_prev = p_estimate.copy()

        # Get current integer coords & check bounds
        current_x_int = int(p_estimate[0] + 0.5)
        current_y_int = int(p_estimate[1] + 0.5)
        if not (r <= current_x_int < frame_width - r and r <= current_y_int < frame_height - r):
             p_estimate = p_estimate_prev
             converged = False
             break

        # Extract current subarray
        y_start, y_end = current_y_int - r, current_y_int + r + 1
        x_start, x_end = current_x_int - r, current_x_int + r + 1
        spot_pixels = image_pixels[y_start:y_end, x_start:x_end]
        rows, cols = spot_pixels.shape

        # Generate coordinate grids and distance squared (using broadcasting)
        x_coords_sub = np.arange(x_start, x_end, dtype=np.float64)
        y_coords_sub = np.arange(y_start, y_end, dtype=np.float64)
        dx = x_coords_sub[np.newaxis, :] - p_estimate[0]
        dy = y_coords_sub[:, np.newaxis] - p_estimate[1]
        dist_sq = dx**2 + dy**2

        # Generate inner circle, Gaussian, and background masks
        inner_mask = (dist_sq <= inner_mask_radius**2).astype(np.uint8)
        exponent = -dist_sq / (2 * gauss_mask_sigma**2)
        gauss_mask = np.exp(exponent)
        gauss_sum = np.sum(gauss_mask)
        gauss_mask = gauss_mask / gauss_sum if gauss_sum != 0.0 else np.zeros_like(gauss_mask)
        bg_mask = 1 - inner_mask

        # Calculate background stats (using loops for Numba compatibility)
        bg_sum = 0.0
        bg_count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if bg_mask[r_idx, c_idx] == 1:
                    bg_sum += spot_pixels[r_idx, c_idx]
                    bg_count += 1

        if bg_count > 0:
            bg_average = bg_sum / bg_count
            # Calculate std dev manually
            bg_sum_sq_diff = 0.0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if bg_mask[r_idx, c_idx] == 1:
                        bg_sum_sq_diff += (spot_pixels[r_idx, c_idx] - bg_average)**2
            bg_std = np.sqrt(bg_sum_sq_diff / bg_count)
        else:
             bg_average = 0.0
             bg_std = 0.0

        # Background correct subarray
        bg_corr_spot_pixels = spot_pixels - bg_average

        # Calculate spot intensity within mask (using loops for Numba)
        spot_intensity = 0.0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if inner_mask[r_idx, c_idx] == 1:
                    spot_intensity += bg_corr_spot_pixels[r_idx, c_idx]

        # Calculate centroid using Gaussian weighted sum
        spot_gaussian_product = bg_corr_spot_pixels * gauss_mask
        sum_spot_gaussian_product = np.sum(spot_gaussian_product)

        p_estimate_new = p_estimate.copy()
        if sum_spot_gaussian_product != 0.0:
            p_estimate_new_0 = np.sum(spot_gaussian_product * x_coords_sub[np.newaxis, :]) / sum_spot_gaussian_product
            p_estimate_new_1 = np.sum(spot_gaussian_product * y_coords_sub[:, np.newaxis]) / sum_spot_gaussian_product
            if np.isnan(p_estimate_new_0) or np.isnan(p_estimate_new_1):
                converged = False; break
            p_estimate_new[0] = p_estimate_new_0
            p_estimate_new[1] = p_estimate_new_1
        else:
             converged = False; break # Avoid division by zero

        # Update estimate and check change
        estimate_change = np.linalg.norm(p_estimate - p_estimate_new)
        p_estimate = p_estimate_new

        # Calculate SNR
        sum_inner_mask = np.sum(inner_mask)
        denominator_snr = bg_std * sum_inner_mask
        snr = abs(spot_intensity / denominator_snr) if denominator_snr != 0.0 else 0.0

        # Check convergence criteria
        if estimate_change < 1e-6: converged = True
        if snr <= snr_filter_cutoff and not converged: break # Stop if SNR too low

    # Return final position, stats, and convergence status
    return (p_estimate, bg_average, spot_intensity, snr, converged)


# --- Main class for storing and processing spots in a frame ---
class Spots:
    # Initialise Spots object
    def __init__(self, num_spots=0, frame=0):
        self.num_spots = num_spots
        self.frame = frame
        if num_spots > 0:
            # Initialise arrays for spot attributes
            self.positions = np.zeros([num_spots, 2]) # x, y
            self.bg_intensity = np.zeros(num_spots)
            self.spot_intensity = np.zeros(num_spots)
            self.traj_num = [-1] * self.num_spots # -1 = unassigned
            self.snr = np.zeros(num_spots)
            self.laser_on_frame = 0 # Unused?
            self.converged = np.zeros(num_spots, dtype=np.int8) # Refinement flag
            self.exists = True
            self.width = np.zeros((num_spots,2)) # width_x, width_y
            self.centre_intensity = np.zeros(num_spots) # Unused?
        else:
            self.exists = False # No spots

    # Set/update spot positions and reset dependent attributes
    def set_positions(self, positions):
        self.num_spots = len(positions)
        # Re-initialise arrays
        self.positions = np.zeros([self.num_spots, 2])
        self.clipping = [False] * self.num_spots # Unused?
        self.bg_intensity = np.zeros(self.num_spots)
        self.spot_intensity = np.zeros(self.num_spots)
        self.width = np.zeros([self.num_spots, 2])
        self.traj_num = [-1] * self.num_spots
        self.snr = np.zeros(self.num_spots)
        self.converged = np.zeros(self.num_spots,dtype=np.int8)
        self.centre_intensity = np.zeros(self.num_spots) # Unused?

        # Fill positions array
        for i in range(self.num_spots):
            self.positions[i, :] = positions[i]

    # Calculate intensity for all spots (parallelisable)
    def get_spot_intensities(self, frame, params):
        if self.num_spots == 0:
            self.spot_intensity = np.array([])
            return

        # Get image data and dimensions
        if hasattr(frame, 'as_image') and callable(frame.as_image):
            image_pixels = frame.as_image()
            frame_size = frame.frame_size
        elif isinstance(frame, np.ndarray):
            image_pixels = frame
            frame_size = (frame.shape[1], frame.shape[0])
        else:
            print("Error: Unsupported frame type in get_spot_intensities.")
            self.spot_intensity = np.zeros(self.num_spots)
            return

        # Check if parallel processing is enabled and safe
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Prepare task arguments for worker func
        task_args = [(self.positions[i], image_pixels, frame_size, params) for i in range(self.num_spots)]

        # Execute intensity calculations
        if run_parallel_internally:
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                results = pool.starmap(_calculate_single_spot_intensity_worker, task_args)
        else:
            results = [_calculate_single_spot_intensity_worker(args) for args in task_args]

        # Store results
        self.spot_intensity = np.array(results)

    # Calculate width for all spots (parallelisable)
    def get_spot_widths(self, frame, params):
        if self.num_spots == 0:
            self.width = np.array([])
            return

        # Get image data and dimensions
        if hasattr(frame, 'as_image') and callable(frame.as_image):
            image_pixels = frame.as_image()
            frame_size = frame.frame_size
        elif isinstance(frame, np.ndarray):
            image_pixels = frame
            frame_size = (frame.shape[1], frame.shape[0])
        else:
            print("Error: Unsupported frame type in get_spot_widths.")
            self.width = np.full((self.num_spots, 2), params.psf_width)
            return

        # Check if parallel processing is enabled and safe
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Prepare task arguments for worker func
        task_args = [(self.positions[i], image_pixels, frame_size, params) for i in range(self.num_spots)]

        # Execute width calculations
        if run_parallel_internally:
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                results = pool.starmap(_calculate_single_spot_width_worker, task_args)
        else:
            results = [_calculate_single_spot_width_worker(args) for args in task_args]

        # Process and store results
        new_widths = np.zeros((self.num_spots, 2))
        success_flags = np.zeros(self.num_spots, dtype=int)
        for i, result in enumerate(results):
            width_xy, success_flag = result
            new_widths[i, :] = width_xy
            success_flags[i] = success_flag
        self.width = new_widths

        # Return success count
        num_success = np.sum(success_flags)
        total_processed = self.num_spots
        return num_success, total_processed

    # Prepare image for spot finding (BGR conversion, blur, kernel)
    def _prepare_image(self, frame: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
        # Convert to BGR
        img_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Create structuring element
        disk_size = 2 * params.struct_disk_radius - 1
        disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))
        # Apply optional Gaussian blur
        if params.filter_image == "Gaussian":
            prepared_frame = cv2.GaussianBlur(img_frame_bgr, (3, 3), 0)
        else:
            prepared_frame = img_frame_bgr.copy()
        return prepared_frame, disk_kernel

    # Calculate threshold for binarisation via histogram analysis
    def _calculate_threshold(self, image: np.ndarray, disk_kernel, params) -> tuple[int, np.ndarray]:
        # Apply Top-Hat transform
        tophatted_frame = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, disk_kernel)
        # Calculate histogram (ignore bin 0)
        hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])
        hist_data[0] = 0
        # Find peak properties
        peak_width, peak_location = algorithms.fwhm(hist_data)
        # Determine threshold
        bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)
        return bw_threshold, tophatted_frame

    # Create binary mask from Top-Hat image using threshold
    def _create_binary_mask(self, tophatted_frame: np.ndarray, threshold: int, params) -> np.ndarray:
        # Blur Top-Hat image
        blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)
        # Threshold
        bw_frame = cv2.threshold(blurred_tophatted_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        # Morphological opening (remove noise)
        bw_opened = cv2.morphologyEx(bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        # Morphological closing (fill holes)
        bw_filled = cv2.morphologyEx(bw_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return bw_filled[:, :, 0] # Return single channel

    # Detect potential spot centres using ultimate erosion
    def _detect_spot_centres(self, binary_mask: np.ndarray) -> np.ndarray:
        # Use ultimate erosion to find peaks
        spot_locations = algorithms.ultimate_erode_scipy(binary_mask)
        # Check for and handle NaNs (unlikely with scipy version)
        if np.isnan(spot_locations).any():
            print("Warning: NaNs detected after ultimate erosion.")
            return np.array([])
        return spot_locations # Returns [x, y] coordinates

    # Find initial spot candidates in a raw frame
    def find_in_frame(self, frame: np.ndarray, params):
        # Prepare image (BGR, blur, kernel)
        prepared_frame, disk_kernel = self._prepare_image(frame, params)
        # Calculate binarisation threshold
        threshold, tophatted_frame = self._calculate_threshold(prepared_frame, disk_kernel, params)
        # Create binary mask (blur, threshold, open, close)
        binary_mask = self._create_binary_mask(tophatted_frame, threshold, params)
        # Detect centres from mask peaks
        spot_locations = self._detect_spot_centres(binary_mask)
        # Store results
        self.set_positions(spot_locations)

    # Merge spots closer than a defined threshold
    def merge_coincident_candidates(self):
        if self.num_spots < 2:
            return # Nothing to merge

        # Set merge distance
        merge_radius = np.sqrt(2.0) # ~1 pixel overlap

        # Build KDTree for efficient neighbour search
        tree = KDTree(self.positions)
        close_pairs = tree.query_pairs(r=merge_radius)

        if not close_pairs:
            return # No pairs close enough

        # Group close pairs into connected components (clusters) using BFS
        adj = collections.defaultdict(list)
        for i, j in close_pairs:
            adj[i].append(j)
            adj[j].append(i)

        clusters = []
        visited = set()
        for i in range(self.num_spots):
            if i not in visited:
                if i in adj: # Part of a close pair
                    current_cluster = set()
                    q = collections.deque([i])
                    visited.add(i)
                    current_cluster.add(i)
                    while q: # BFS traversal
                        u = q.popleft()
                        for v in adj[u]:
                            if v not in visited:
                                visited.add(v)
                                current_cluster.add(v)
                                q.append(v)
                    clusters.append(list(current_cluster))
                else: # Isolated spot
                    clusters.append([i])
                    visited.add(i)

        # Calculate new positions: centroid for multi-spot clusters, original for single spots
        new_positions = []
        for cluster_indices in clusters:
            if len(cluster_indices) == 1:
                new_positions.append(self.positions[cluster_indices[0]])
            else:
                cluster_pos = self.positions[cluster_indices]
                mean_pos = np.mean(cluster_pos, axis=0) # Centroid
                new_positions.append(mean_pos)

        # Update object with merged positions
        if not new_positions:
            print("Warning: Merging resulted in no spots left.")
            self.set_positions(np.array([]))
        else:
            self.set_positions(np.array(new_positions))

    # Filter spots based on SNR, proximity to edge, and optional mask
    def filter_candidates(self, frame, params):
        if self.num_spots == 0:
            return

        sys.stdout.flush() # Flush buffer for potential logs

        keep_mask = np.ones(self.num_spots, dtype=bool) # Start with all True
        initial_indices = np.arange(self.num_spots) # For potential logging

        # Filter by SNR
        snr_values = self.snr
        snr_ok = snr_values > params.snr_filter_cutoff
        keep_mask &= snr_ok
        sys.stdout.flush() # Flush buffer

        # Filter by edge proximity
        frame_height, frame_width = frame.frame_size[1], frame.frame_size[0]
        hw = params.subarray_halfwidth
        positions_to_check = self.positions
        too_close_x = (positions_to_check[:, 0] < hw) | (positions_to_check[:, 0] >= frame_width - hw)
        too_close_y = (positions_to_check[:, 1] < hw) | (positions_to_check[:, 1] >= frame_height - hw)
        edges_ok = ~(too_close_x | too_close_y)
        keep_mask &= edges_ok
        sys.stdout.flush() # Flush buffer

        # Filter by mask (if frame has one)
        if frame.has_mask:
            rounded_y = np.clip(np.round(positions_to_check[:, 1]).astype(int), 0, frame_height - 1)
            rounded_x = np.clip(np.round(positions_to_check[:, 0]).astype(int), 0, frame_width - 1)
            try:
                mask_values = frame.mask_data[rounded_y, rounded_x]
                mask_ok = (mask_values != 0) # Keep if mask value is non-zero
                keep_mask &= mask_ok
                sys.stdout.flush() # Flush buffer
            except IndexError:
                 pass # Ignore if index error occurs (should be rare with clip)

        # Apply the combined filter mask to all relevant attributes
        final_kept_count = np.sum(keep_mask)
        if final_kept_count < self.num_spots:
            self.positions = self.positions[keep_mask]
            self.bg_intensity = self.bg_intensity[keep_mask]
            self.spot_intensity = self.spot_intensity[keep_mask]
            self.width = self.width[keep_mask]
             # Convert list to array for boolean indexing, then back to list
            self.traj_num = list(np.array(self.traj_num)[keep_mask])
            self.snr = self.snr[keep_mask]
            self.converged = self.converged[keep_mask]
            # Check other attributes if they exist and need filtering
            if hasattr(self, 'clipping'):
                self.clipping = list(np.array(self.clipping)[keep_mask])
            if hasattr(self, 'centre_intensity'):
                 self.centre_intensity = self.centre_intensity[keep_mask]

        # Update spot count
        self.num_spots = final_kept_count
        sys.stdout.flush() # Flush buffer

    # Calculate pairwise distances between spots in this frame and another
    def distance_from(self, other):
        # Create distance matrix [self.num_spots x other.num_spots]
        distances = np.zeros([self.num_spots, other.num_spots])
        # Fill matrix with Euclidean distances
        for i in range(self.num_spots):
            for j in range(other.num_spots):
                distances[i, j] = np.linalg.norm(self.positions[i, :] - other.positions[j, :])
        return distances

    # Assign initial trajectory numbers (0 to N-1) for the first frame
    def index_first(self):
        self.traj_num = list(range(self.num_spots))

    # Link current spots to previous spots based on proximity
    def link(self, prev_spots, params):
        # Calculate all pairwise distances
        distances = self.distance_from(prev_spots)
        # Get indices of previous spots sorted by distance for each current spot
        neighbours = np.argsort(distances[:, :], axis=1)
        # Track already linked previous spots
        paired_spots = []
        # Determine next available trajectory ID
        next_trajectory = max(prev_spots.traj_num) + 1 if prev_spots.num_spots > 0 else 0

        # Iterate through current spots
        for i in range(self.num_spots):
            linked = False
            # Check neighbours in order of proximity
            for j in range(prev_spots.num_spots):
                neighbour_idx = neighbours[i, j]
                # Check distance threshold
                if distances[i, neighbour_idx] < params.max_displacement:
                    # Check if neighbour already linked
                    if neighbour_idx in paired_spots:
                        continue # Try next neighbour
                    else:
                        # Link current spot to this neighbour
                        paired_spots.append(neighbour_idx)
                        self.traj_num[i] = prev_spots.traj_num[neighbour_idx]
                        linked = True
                        break # Stop searching neighbours for spot i
                else:
                    # Nearest available neighbour is too far, assign new trajectory
                    self.traj_num[i] = next_trajectory
                    next_trajectory += 1
                    linked = True
                    break # Stop searching neighbours for spot i

            # Handle case where prev_spots was empty
            if not linked and prev_spots.num_spots == 0:
                 self.traj_num[i] = next_trajectory
                 next_trajectory += 1
                 linked = True

            # Sanity check: ensure spot was assigned a trajectory
            if self.traj_num[i] == -1:
                sys.exit(f"ERROR: Failed linking for spot {i}, frame {self.frame}")

    # Refine spot centres to sub-pixel precision (parallelisable)
    def refine_centres(self, frame, params):
        if self.num_spots == 0:
            return

        # Get image data and dimensions
        image_pixels = frame.as_image()
        frame_size_tuple = tuple(frame.frame_size)
        # Check if parallel processing is enabled and safe
        run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

        # Bundle required params for Numba worker func
        params_tuple = (
            params.subarray_halfwidth,
            params.inner_mask_radius,
            params.gauss_mask_sigma,
            params.gauss_mask_max_iter,
            params.snr_filter_cutoff
        )

        # Prepare task arguments
        task_args = [(self.positions[i], image_pixels, frame_size_tuple, *params_tuple) for i in range(self.num_spots)]

        # Execute refinement calculations
        if run_parallel_internally:
            with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
                 results = pool.map(_refine_single_spot_worker, task_args) # map for single tuple arg
        else:
            results = [_refine_single_spot_worker(args) for args in task_args]

        # Unpack results and update attributes
        new_positions = np.zeros_like(self.positions)
        new_bg_intensity = np.zeros_like(self.bg_intensity)
        new_spot_intensity = np.zeros_like(self.spot_intensity)
        new_snr = np.zeros_like(self.snr)
        new_converged = np.zeros_like(self.converged, dtype=bool)

        for i, result in enumerate(results):
            final_pos, bg_avg, spot_int, snr_val, conv_status = result
            new_positions[i, :] = final_pos
            new_bg_intensity[i] = bg_avg
            new_spot_intensity[i] = spot_int
            new_snr[i] = snr_val
            new_converged[i] = conv_status

        # Update object attributes with refined values
        self.positions = new_positions
        self.bg_intensity = new_bg_intensity
        self.spot_intensity = new_spot_intensity
        self.snr = new_snr
        self.converged = new_converged.astype(np.int8) # Convert bool back to int8

# """ SPOTS - Spot characterisation and manipulation module

# Description:
#     spots.py contains the Spots class for characterising and manipulating data
#     associated with bright spots in the image datasets provided.

# Contains:
#     class Spots

# Author:
#     Edward Higgins

# Version: 0.2.1
# """

# import sys

# import cv2
# #import matplotlib.pyplot as plt
# import numpy as np
# # Import local algorithms module
# from . import algorithms
# # Import KDTree for efficient neighbour searches
# from scipy.spatial import KDTree
# # Import collections for defaultdict and deque (used in merging)
# import collections
# # Import multiprocessing utilities
# import multiprocessing
# import contextlib
# from numba import jit

# # --- Worker function for calculating intensity of a single spot ---
# def _calculate_single_spot_intensity_worker(args):
#     # Unpack arguments: position, pixel data, frame dimensions, parameters
#     position, image_pixels, frame_size, params = args
#     frame_width, frame_height = frame_size
#     # Get subarray radius from parameters
#     r = params.subarray_halfwidth

#     # Round position to integer indices for array slicing
#     x_int, y_int = round(position[0]), round(position[1])

#     # Check if spot is too close to edge for subarray extraction
#     if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
#         # Return 0 if spot is near edge
#         return 0.0

#     # Define subarray boundaries
#     y_start, y_end = y_int - r, y_int + r + 1
#     x_start, x_end = x_int - r, x_int + r + 1
#     # Extract the subarray (region around the spot)
#     tmp = image_pixels[y_start:y_end, x_start:x_end]

#     # Create a circular mask centred in the subarray
#     spotmask = np.zeros(tmp.shape, dtype=np.uint8)
#     # Draw filled circle (value 1) within the mask
#     cv2.circle(spotmask,
#                (r, r), # Centre coordinates (r, r) in the subarray
#                params.inner_mask_radius, # Radius from params
#                1,    # Mask value (integer 1)
#                -1)   # Fill the circle

#     # Calculate background intensity from pixels *outside* the mask
#     bg_pixels = tmp[spotmask == 0]
#     # Check if any background pixels exist
#     if bg_pixels.size > 0:
#         bgintensity = np.mean(bg_pixels)
#     else:
#         # Default background to 0 if mask covers entire subarray
#         bgintensity = 0

#     # Subtract background from the subarray
#     tmp_bg_corr = tmp - bgintensity
#     # Calculate spot intensity by summing pixels *inside* the mask
#     intensity = np.sum(tmp_bg_corr[spotmask == 1])

#     # Return the calculated background-corrected intensity
#     return intensity


# # --- Worker function for calculating width of a single spot ---
# def _calculate_single_spot_width_worker(args):
#     # Unpack arguments: position, pixel data, frame dimensions, parameters
#     position, image_pixels, frame_size, params = args
#     frame_width, frame_height = frame_size
#     # Get subarray radius and fallback width from parameters
#     r = params.subarray_halfwidth
#     fallback_width = params.psf_width # Default if fit fails

#     # Round position to integer indices
#     x_int, y_int = round(position[0]), round(position[1])

#     # Check if spot is too close to edge
#     if not (r <= x_int < frame_width - r and r <= y_int < frame_height - r):
#         # Return fallback width and failure flag if near edge
#         return ([fallback_width, fallback_width], 0)

#     # Define subarray boundaries
#     y_start, y_end = y_int - r, y_int + r + 1
#     x_start, x_end = x_int - r, x_int + r + 1
#     # Extract subarray
#     tmp = image_pixels[y_start:y_end, x_start:x_end]

#     # Create circular mask (needed for background calculation)
#     spotmask = np.zeros(tmp.shape, dtype=np.uint8)
#     cv2.circle(spotmask, (r, r), params.inner_mask_radius, 1, -1)

#     # Calculate background intensity from outside the mask
#     bg_pixels = tmp[spotmask == 0]
#     if bg_pixels.size > 0:
#         bgintensity = np.mean(bg_pixels)
#     else:
#         bgintensity = 0

#     # Subtract background from subarray
#     tmp_bg_corr = tmp - bgintensity

#     # Fit 2D Gaussian to the background-corrected subarray
#     p, succ = algorithms.fit2Dgaussian(tmp_bg_corr)

#     # Check if fit succeeded and widths (p[3], p[4]) are positive
#     if succ == 1 and p[3] > 0 and p[4] > 0:
#         # Store fitted widths [wx, wy]
#         width_xy = [p[3], p[4]]
#         success_flag = 1 # Indicate success
#     else:
#         # Use fallback widths if fit failed or widths non-physical
#         width_xy = [fallback_width, fallback_width]
#         success_flag = 0 # Indicate failure

#     # Return calculated/fallback widths and success flag
#     return (width_xy, success_flag)

# # --- Worker function for refining centre of a single spot ---
# @jit(nopython=True)
# def _refine_single_spot_worker(args):
#     """
#     Worker function to refine a single spot's center. Numba JIT applied.
#     Uses broadcasting instead of meshgrid and explicit loops for boolean indexing.
#     """
#     # --- Unpack arguments ---
#     initial_position, image_pixels, frame_size_tuple, \
#     subarray_halfwidth, inner_mask_radius, gauss_mask_sigma, \
#     gauss_mask_max_iter, snr_filter_cutoff = args
#     # --- End unpacking ---

#     p_estimate = initial_position.copy()
#     frame_width, frame_height = frame_size_tuple
#     r = subarray_halfwidth

#     # --- Initial Checks ---
#     spot_failed = False
#     for d in (0, 1):
#         p_estimate_d_int = int(p_estimate[d] + 0.5)
#         if p_estimate_d_int < r or p_estimate_d_int >= frame_size_tuple[d] - r:
#             spot_failed = True
#             break
#     if spot_failed:
#         return (initial_position, 0.0, 0.0, 0.0, False)

#     # --- Iterative Refinement Loop ---
#     converged = False
#     iteration = 0
#     bg_average = 0.0
#     spot_intensity = 0.0
#     snr = 0.0
#     p_estimate_prev = p_estimate.copy()

#     while not converged and iteration < gauss_mask_max_iter:
#         iteration += 1

#         # --- Oscillation / NaN Check ---
#         if iteration > 5 and np.linalg.norm(p_estimate - p_estimate_prev) < 1e-9 and not converged: break
#         if np.any(np.isnan(p_estimate)):
#             p_estimate = p_estimate_prev
#             converged = False
#             break
#         p_estimate_prev = p_estimate.copy()
#         # --- End check ---

#         current_x_int = int(p_estimate[0] + 0.5)
#         current_y_int = int(p_estimate[1] + 0.5)

#         if not (r <= current_x_int < frame_width - r and r <= current_y_int < frame_height - r):
#              p_estimate = p_estimate_prev
#              converged = False
#              break

#         y_start, y_end = current_y_int - r, current_y_int + r + 1
#         x_start, x_end = current_x_int - r, current_x_int + r + 1

#         spot_pixels = image_pixels[y_start:y_end, x_start:x_end]
#         rows, cols = spot_pixels.shape # Get dimensions for looping

#         # --- Coordinate Generation (Broadcasting) ---
#         x_coords_sub = np.arange(x_start, x_end, dtype=np.float64)
#         y_coords_sub = np.arange(y_start, y_end, dtype=np.float64)
#         dx = x_coords_sub[np.newaxis, :] - p_estimate[0]
#         dy = y_coords_sub[:, np.newaxis] - p_estimate[1]
#         dist_sq = dx**2 + dy**2
#         # --- End Coordinate Generation ---

#         # --- Mask Generation ---
#         inner_mask = (dist_sq <= inner_mask_radius**2).astype(np.uint8)
#         exponent = -dist_sq / (2 * gauss_mask_sigma**2)
#         gauss_mask = np.exp(exponent)
#         gauss_sum = np.sum(gauss_mask)
#         if gauss_sum != 0.0:
#             gauss_mask /= gauss_sum
#         else:
#             gauss_mask = np.zeros_like(gauss_mask)
#         bg_mask = 1 - inner_mask
#         # --- End Mask Generation ---

#         # ***** START BOOLEAN INDEXING REPLACEMENT *****
#         # --- Calculate Background Stats using Loops ---
#         bg_sum = 0.0
#         bg_count = 0
#         for r_idx in range(rows):
#             for c_idx in range(cols):
#                 if bg_mask[r_idx, c_idx] == 1:
#                     bg_sum += spot_pixels[r_idx, c_idx]
#                     bg_count += 1

#         if bg_count > 0:
#             bg_average = bg_sum / bg_count
#             # Calculate std dev manually for Numba compatibility if np.std fails
#             bg_sum_sq_diff = 0.0
#             for r_idx in range(rows):
#                 for c_idx in range(cols):
#                     if bg_mask[r_idx, c_idx] == 1:
#                         bg_sum_sq_diff += (spot_pixels[r_idx, c_idx] - bg_average)**2
#             bg_std = np.sqrt(bg_sum_sq_diff / bg_count)
#         else:
#              bg_average = 0.0
#              bg_std = 0.0

#         # --- Calculate bg_corr_spot_pixels (still fast) ---
#         bg_corr_spot_pixels = spot_pixels - bg_average

#         # --- Calculate Spot Intensity using Loops ---
#         spot_intensity = 0.0
#         for r_idx in range(rows):
#             for c_idx in range(cols):
#                 if inner_mask[r_idx, c_idx] == 1:
#                     spot_intensity += bg_corr_spot_pixels[r_idx, c_idx]
#         # ***** END BOOLEAN INDEXING REPLACEMENT *****

#         # --- Calculate Centroid (remains the same) ---
#         spot_gaussian_product = bg_corr_spot_pixels * gauss_mask
#         sum_spot_gaussian_product = np.sum(spot_gaussian_product)

#         p_estimate_new = p_estimate.copy()
#         if sum_spot_gaussian_product != 0.0:
#             p_estimate_new_0 = np.sum(spot_gaussian_product * x_coords_sub[np.newaxis, :]) / sum_spot_gaussian_product
#             p_estimate_new_1 = np.sum(spot_gaussian_product * y_coords_sub[:, np.newaxis]) / sum_spot_gaussian_product
#             if np.isnan(p_estimate_new_0) or np.isnan(p_estimate_new_1):
#                 converged = False; break
#             p_estimate_new[0] = p_estimate_new_0
#             p_estimate_new[1] = p_estimate_new_1
#         else:
#              converged = False; break

#         estimate_change = np.linalg.norm(p_estimate - p_estimate_new)
#         p_estimate = p_estimate_new

#         # --- Calculate SNR ---
#         sum_inner_mask = np.sum(inner_mask) # Sum of mask (number of pixels)
#         denominator_snr = bg_std * sum_inner_mask
#         if denominator_snr != 0.0:
#              snr = abs(spot_intensity / denominator_snr)
#         else:
#              snr = 0.0

#         # --- Convergence Checks ---
#         if estimate_change < 1e-6: converged = True
#         if snr <= snr_filter_cutoff and not converged: break
#     # --- End of loop ---

#     return (p_estimate, bg_average, spot_intensity, snr, converged)

# # Class definition for storing and manipulating spot data within a frame
# class Spots:
#     # Initialiser for the Spots object
#     def __init__(self, num_spots=0, frame=0):
#         # Store number of spots and frame number
#         self.num_spots = num_spots
#         self.frame = frame
#         # If num_spots > 0, initialise arrays for spot attributes
#         if num_spots > 0:
#             self.positions = np.zeros([num_spots, 2]) # [x, y] coordinates
#             self.bg_intensity = np.zeros(num_spots) # Background intensity
#             self.spot_intensity = np.zeros(num_spots) # Spot intensity
#             self.traj_num = [-1] * self.num_spots # Trajectory ID (-1 initially)
#             self.snr = np.zeros(num_spots) # Signal-to-noise ratio
#             self.laser_on_frame = 0 # Placeholder (seems unused here)
#             self.converged = np.zeros(num_spots, dtype=np.int8) # Refinement convergence flag
#             self.exists = True # Flag indicating object contains data
#             self.width = np.zeros((num_spots,2)) # Spot width [width_x, width_y]
#             self.centre_intensity = np.zeros(num_spots) # Intensity at centre pixel (unused?)
#         else:
#             # If num_spots is 0, set exists flag to False
#             self.exists = False

#     # Method to set/update spot positions and reset related attributes
#     def set_positions(self, positions):
#         # Update number of spots based on input positions array
#         self.num_spots = len(positions)
#         # Re-initialise all attribute arrays based on the new number of spots
#         self.positions = np.zeros([self.num_spots, 2])
#         self.clipping = [False] * self.num_spots # Clipping flag (unused?)
#         self.bg_intensity = np.zeros(self.num_spots)
#         self.spot_intensity = np.zeros(self.num_spots)
#         self.width = np.zeros([self.num_spots, 2])
#         self.traj_num = [-1] * self.num_spots
#         self.snr = np.zeros(self.num_spots)
#         self.converged = np.zeros(self.num_spots,dtype=np.int8)
#         self.centre_intensity = np.zeros(self.num_spots) # (unused?)

#         # Populate the positions array with the input data
#         for i in range(self.num_spots):
#             self.positions[i, :] = positions[i]

#     # Method to calculate background-corrected spot intensities
#     def get_spot_intensities(self, frame, params):
#         # Check if there are any spots to process
#         if self.num_spots == 0:
#             self.spot_intensity = np.array([]) # Ensure attribute exists
#             return

#         # Get pixel data and frame size, handling both ImageData and ndarray inputs
#         if hasattr(frame, 'as_image') and callable(frame.as_image):
#             image_pixels = frame.as_image()
#             frame_size = frame.frame_size
#         elif isinstance(frame, np.ndarray):
#             image_pixels = frame
#             frame_size = (frame.shape[1], frame.shape[0]) # Assume (height, width) -> (width, height)
#         else:
#             print("Error: Unsupported frame data type in get_spot_intensities.")
#             self.spot_intensity = np.zeros(self.num_spots) # Assign default
#             return

#         # Determine if multiprocessing should be used internally
#         # Only run in parallel if requested AND not already in a worker process
#         run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

#         # Prepare arguments for each spot calculation task
#         task_args = []
#         for i in range(self.num_spots):
#             task_args.append((self.positions[i], image_pixels, frame_size, params))

#         # Execute tasks either in parallel or serially
#         if run_parallel_internally:
#             # Use multiprocessing pool
#             with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
#                 results = pool.starmap(_calculate_single_spot_intensity_worker, task_args)
#         else:
#             # Run serially
#             results = []
#             for args in task_args:
#                 results.append(_calculate_single_spot_intensity_worker(args))

#         # Store the calculated intensities in the object's attribute
#         self.spot_intensity = np.array(results)

#     # Method to calculate spot widths using 2D Gaussian fitting
#     def get_spot_widths(self, frame, params):
#         # Check if there are spots to process
#         if self.num_spots == 0:
#             self.width = np.array([]) # Ensure attribute exists
#             return

#         # Get pixel data and frame size
#         if hasattr(frame, 'as_image') and callable(frame.as_image):
#             image_pixels = frame.as_image()
#             frame_size = frame.frame_size
#         elif isinstance(frame, np.ndarray):
#             image_pixels = frame
#             frame_size = (frame.shape[1], frame.shape[0]) # Assume (height, width) -> (width, height)
#         else:
#             print("Error: Unsupported frame data type in get_spot_widths.")
#             # Assign default widths based on psf_width parameter
#             self.width = np.full((self.num_spots, 2), params.psf_width)
#             return

#         # Determine if multiprocessing should be used
#         run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

#         # Prepare arguments for each spot width calculation
#         task_args = []
#         for i in range(self.num_spots):
#             task_args.append((self.positions[i], image_pixels, frame_size, params))

#         # Execute tasks in parallel or serially
#         if run_parallel_internally:
#             with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
#                 results = pool.starmap(_calculate_single_spot_width_worker, task_args)
#         else:
#             results = []
#             for args in task_args:
#                 results.append(_calculate_single_spot_width_worker(args))

#         # --- Process results and update attributes ---
#         new_widths = np.zeros((self.num_spots, 2))
#         success_flags = np.zeros(self.num_spots, dtype=int)

#         # Unpack results (width_xy, success_flag) for each spot
#         for i, result in enumerate(results):
#             width_xy, success_flag = result
#             new_widths[i, :] = width_xy
#             success_flags[i] = success_flag

#         # Store the calculated/fallback widths
#         self.width = new_widths

#         # Calculate and return the number of successful fits and total processed
#         num_success = np.sum(success_flags)
#         total_processed = self.num_spots
#         return num_success, total_processed

#     # --- Helper method for image preparation ---
#     def _prepare_image(self, frame: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
#         # Convert grayscale frame to BGR (required by some OpenCV functions)
#         img_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         # Create elliptical structuring element (disk) for morphological operations
#         disk_size = 2 * params.struct_disk_radius - 1 # Diameter
#         disk_kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (disk_size, disk_size)
#         )

#         # Apply Gaussian blur if specified in parameters
#         if params.filter_image == "Gaussian":
#             prepared_frame = cv2.GaussianBlur(img_frame_bgr, (3, 3), 0) # 3x3 kernel
#         else:
#             # Otherwise, use the original BGR frame
#             prepared_frame = img_frame_bgr.copy()

#         # Return the prepared frame and the structuring element
#         return prepared_frame, disk_kernel

#     # --- Helper method for threshold calculation ---
#     def _calculate_threshold(self, image: np.ndarray, disk_kernel, params) -> tuple[int, np.ndarray]:
#         # Apply morphological Top-Hat filter to enhance bright spots on dark background
#         tophatted_frame = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, disk_kernel)

#         # Calculate histogram of the top-hat filtered image (channel 0)
#         hist_data = cv2.calcHist([tophatted_frame], [0], None, [256], [0, 256])
#         # Ignore the first bin (often background noise)
#         hist_data[0] = 0

#         # Calculate Full Width at Half Maximum (FWHM) and peak location of histogram
#         peak_width, peak_location = algorithms.fwhm(hist_data)
#         # Determine binary threshold based on peak location and width tolerance
#         bw_threshold = int(peak_location + params.bw_threshold_tolerance * peak_width)

#         # Return the calculated threshold and the top-hat image
#         return bw_threshold, tophatted_frame

#     # --- Helper method for creating binary mask ---
#     def _create_binary_mask(self, tophatted_frame: np.ndarray, threshold: int, params) -> np.ndarray:
#         # Apply Gaussian blur to the top-hat image to smooth noise
#         blurred_tophatted_frame = cv2.GaussianBlur(tophatted_frame, (3, 3), 0)

#         # Apply binary thresholding using the calculated threshold
#         bw_frame = cv2.threshold(
#             blurred_tophatted_frame, threshold, 255, cv2.THRESH_BINARY
#         )[1] # Get the thresholded image (second element of tuple)

#         # Apply morphological Opening (erosion then dilation) to remove small noise pixels
#         bw_opened = cv2.morphologyEx(
#             bw_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # Use 3x3 cross kernel
#         )

#         # Apply morphological Closing (dilation then erosion) to fill small holes in spots
#         bw_filled = cv2.morphologyEx(
#             bw_opened,
#             cv2.MORPH_CLOSE,
#             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), # Use 3x3 elliptical kernel
#         )
#         # Return the first channel of the result (it's binary/grayscale)
#         return bw_filled[:, :, 0]

#     # --- Helper method for detecting spot centres ---
#     def _detect_spot_centres(self, binary_mask: np.ndarray) -> np.ndarray:
#         # Use ultimate erosion algorithm (find peaks in distance transform)
#         spot_locations = algorithms.ultimate_erode_scipy(binary_mask)
#         # Check for NaN values in the result (shouldn't happen with scipy version)
#         if np.isnan(spot_locations).any():
#             print("Warning: NaNs detected after ultimate erosion.")
#             # Return empty array if NaNs found
#             return np.array([])
#         # Return array of [x, y] coordinates
#         return spot_locations

#     # --- Main public method for finding spots in a frame ---
#     def find_in_frame(self, frame: np.ndarray, params):
#         # 1. Image Preparation: Convert to BGR, apply blur, create kernel
#         prepared_frame, disk_kernel = self._prepare_image(frame, params)

#         # 2. Threshold Calculation: Apply top-hat, analyse histogram
#         threshold, tophatted_frame = self._calculate_threshold(prepared_frame, disk_kernel, params)

#         # 3. Binary Mask Creation: Blur, threshold, open, close
#         binary_mask = self._create_binary_mask(tophatted_frame, threshold, params)

#         # 4. Spot Centre Detection: Use ultimate erosion on the binary mask
#         spot_locations = self._detect_spot_centres(binary_mask)

#         # 5. Store Results: Update the object's positions attribute
#         self.set_positions(spot_locations)

#     # Method to merge spots that are very close together
#     def merge_coincident_candidates(self):
#         # Skip if fewer than 2 spots exist
#         if self.num_spots < 2:
#             return

#         # Define merge distance threshold (sqrt(2) pixels)
#         merge_radius = np.sqrt(2.0)

#         # Build KDTree for efficient nearest neighbour search
#         tree = KDTree(self.positions)

#         # Find all pairs of spots closer than merge_radius
#         close_pairs = tree.query_pairs(r=merge_radius)

#         # Skip if no pairs are close enough
#         if not close_pairs:
#             return

#         # --- Group close pairs into clusters using graph connectivity (connected components) ---
#         # Build adjacency list representation of the graph where nodes are spot indices
#         adj = collections.defaultdict(list)
#         for i, j in close_pairs:
#             adj[i].append(j)
#             adj[j].append(i)

#         # Find connected components (clusters) using Breadth-First Search (BFS)
#         clusters = [] # List to store clusters (each cluster is a list of spot indices)
#         visited = set() # Keep track of visited spot indices

#         # Iterate through all spot indices
#         for i in range(self.num_spots):
#             # If spot 'i' is part of a close pair and not yet visited, start BFS
#             if i not in visited and i in adj:
#                 current_cluster = set() # Store indices in the current cluster
#                 q = collections.deque([i]) # Queue for BFS
#                 visited.add(i)
#                 current_cluster.add(i)

#                 # Perform BFS
#                 while q:
#                     u = q.popleft() # Get current node
#                     # Explore neighbours
#                     for v in adj[u]:
#                         if v not in visited:
#                             visited.add(v)
#                             current_cluster.add(v)
#                             q.append(v)
#                 # Add the found cluster to the list
#                 clusters.append(list(current_cluster))
#             # If spot 'i' was not part of any close pair and not visited, add as a single-spot cluster
#             elif i not in visited and i not in adj:
#                 clusters.append([i])
#                 visited.add(i) # Mark as visited
#         # --- End of cluster finding ---

#         # --- Calculate new positions based on cluster centroids ---
#         new_positions = []
#         # Iterate through the found clusters
#         for cluster_indices in clusters:
#             # If cluster has only one spot, keep original position
#             if len(cluster_indices) == 1:
#                 spot_index = cluster_indices[0]
#                 new_positions.append(self.positions[spot_index])
#             # If cluster has multiple spots, calculate centroid
#             else:
#                 cluster_pos = self.positions[cluster_indices] # Get positions of spots in cluster
#                 mean_pos = np.mean(cluster_pos, axis=0) # Calculate mean position (centroid)
#                 new_positions.append(mean_pos)

#         # Update the object's positions with the new (potentially merged) positions
#         if not new_positions:
#              # Handle unlikely case where merging leaves no spots
#             print("Warning: Merging resulted in no spots left.")
#             self.set_positions(np.array([]))
#         else:
#             self.set_positions(np.array(new_positions))

#     def filter_candidates(self, frame, params):
#         """
#         Filters spot candidates based on SNR, proximity to edges, and optional mask.
#         """
#         if self.num_spots == 0:
#             #print(f"Frame {self.frame}: filter_candidates called with 0 spots.") # Added print
#             return

#         # ***** ADD DETAIL PRINT *****
#         #print(f"Frame {self.frame}: filter_candidates START - Num spots: {self.num_spots}")
#         sys.stdout.flush()
#         # ***** END DETAIL PRINT *****

#         keep_mask = np.ones(self.num_spots, dtype=bool)
#         initial_indices = np.arange(self.num_spots) # Keep track of original index for logging

#         # 1. Filter by SNR
#         snr_values = self.snr # Get SNR array
#         snr_ok = snr_values > params.snr_filter_cutoff
#         keep_mask &= snr_ok
#         # ***** ADD DETAIL PRINT *****
#         filtered_snr = initial_indices[~snr_ok]
#         if len(filtered_snr) > 0:
#              #print(f"Frame {self.frame}: filter_candidates - Filtered {len(filtered_snr)} by SNR (<={params.snr_filter_cutoff}). Indices: {filtered_snr}")
#              #print(f"  - Corresponding SNR values: {snr_values[~snr_ok]}")
#              sys.stdout.flush()
#         # ***** END DETAIL PRINT *****


#         # 2. Filter by proximity to edges
#         frame_height, frame_width = frame.frame_size[1], frame.frame_size[0]
#         hw = params.subarray_halfwidth
#         positions_to_check = self.positions # Get positions array

#         too_close_x = (positions_to_check[:, 0] < hw) | (positions_to_check[:, 0] >= frame_width - hw)
#         too_close_y = (positions_to_check[:, 1] < hw) | (positions_to_check[:, 1] >= frame_height - hw)
#         edges_ok = ~(too_close_x | too_close_y)
#         keep_mask &= edges_ok
#         # ***** ADD DETAIL PRINT *****
#         filtered_edge = initial_indices[~edges_ok & snr_ok] # Log only those filtered here, not already by SNR
#         if len(filtered_edge) > 0:
#              #print(f"Frame {self.frame}: filter_candidates - Filtered {len(filtered_edge)} by Edges. Indices: {filtered_edge}")
#              #print(f"  - Corresponding positions: {positions_to_check[~edges_ok & snr_ok]}")
#              sys.stdout.flush()
#         # ***** END DETAIL PRINT *****

#         # 3. Filter by mask (if applicable)
#         if frame.has_mask:
#             rounded_y = np.clip(np.round(positions_to_check[:, 1]).astype(int), 0, frame_height - 1)
#             rounded_x = np.clip(np.round(positions_to_check[:, 0]).astype(int), 0, frame_width - 1)
#             try:
#                 mask_values = frame.mask_data[rounded_y, rounded_x]
#                 mask_ok = (mask_values != 0)
#                 keep_mask &= mask_ok
#                 # ***** ADD DETAIL PRINT *****
#                 filtered_mask = initial_indices[~mask_ok & edges_ok & snr_ok] # Log only those filtered here
#                 if len(filtered_mask) > 0:
#                      #print(f"Frame {self.frame}: filter_candidates - Filtered {len(filtered_mask)} by Mask. Indices: {filtered_mask}")
#                      sys.stdout.flush()
#                 # ***** END DETAIL PRINT *****
#             except IndexError:
#                 #print(f"Warning: Frame {self.frame} - Index error during mask filtering.")
#                 pass


#         # Apply the final mask
#         final_kept_count = np.sum(keep_mask)
#         # ***** ADD DETAIL PRINT *****
#         #print(f"Frame {self.frame}: filter_candidates END - Keeping {final_kept_count} / {self.num_spots} spots.")
#         sys.stdout.flush()
#         # ***** END DETAIL PRINT *****

#         # --- Apply the final mask to all relevant attributes ---
#         # ... (rest of filtering code) ...
#         self.num_spots = final_kept_count # Use count derived from mask

#     # Method to calculate pairwise distances between spots in this object and another Spots object
#     def distance_from(self, other):
#         # Initialise matrix to store distances [num_spots_self, num_spots_other]
#         distances = np.zeros([self.num_spots, other.num_spots])

#         # Calculate Euclidean distance for each pair of spots
#         for i in range(self.num_spots):
#             for j in range(other.num_spots):
#                 # Distance = sqrt( (x1-x2)^2 + (y1-y2)^2 )
#                 distances[i, j] = np.linalg.norm(
#                     self.positions[i, :] - other.positions[j, :]
#                 )
#         # Return the distance matrix
#         return distances

#     # Method to assign initial trajectory numbers (0 to num_spots-1) - typically used for the first frame
#     def index_first(self):
#         self.traj_num = list(range(self.num_spots))

#     # Method to link spots in the current frame to spots in the previous frame
#     def link(self, prev_spots, params):
#         # Calculate distances between all current spots and all previous spots
#         distances = self.distance_from(prev_spots)

#         # Find the indices of previous spots sorted by distance for each current spot
#         neighbours = np.argsort(distances[:, :], axis=1)
#         # Keep track of previous spots that have already been linked
#         paired_spots = []
#         # Determine the next available trajectory ID
#         next_trajectory = max(prev_spots.traj_num) + 1 if prev_spots.num_spots > 0 else 0 # Handle empty prev_spots

#         # Iterate through each spot in the current frame
#         for i in range(self.num_spots):
#             linked = False # Flag to check if spot 'i' gets linked
#             # Iterate through potential matches (nearest previous spots first)
#             for j in range(prev_spots.num_spots):
#                 neighbour_idx = neighbours[i, j] # Index of the j-th nearest previous spot
#                 # Check if the distance to this neighbour is within the maximum allowed displacement
#                 if distances[i, neighbour_idx] < params.max_displacement:
#                     # Check if this neighbour has already been paired with another current spot
#                     if neighbour_idx in paired_spots:
#                         # If already paired, continue to the next nearest neighbour
#                         continue
#                     else:
#                         # If not paired, link spot 'i' to this neighbour
#                         paired_spots.append(neighbour_idx) # Mark neighbour as paired
#                         self.traj_num[i] = prev_spots.traj_num[neighbour_idx] # Assign same trajectory ID
#                         linked = True # Mark spot 'i' as linked
#                         break # Move to the next current spot 'i'
#                 else:
#                     # If the nearest unpaired neighbour is too far, assign a new trajectory ID
#                     self.traj_num[i] = next_trajectory
#                     next_trajectory += 1 # Increment ID for the next new trajectory
#                     linked = True # Mark spot 'i' as linked (to a new trajectory)
#                     break # Move to the next current spot 'i'

#             # If after checking all neighbours, the spot wasn't linked (should only happen if prev_spots is empty)
#             if not linked and prev_spots.num_spots == 0:
#                  self.traj_num[i] = next_trajectory
#                  next_trajectory += 1
#                  linked = True


#             # Error check: Ensure every spot gets assigned a trajectory ID (should always happen)
#             if self.traj_num[i] == -1:
#                 sys.exit(f"ERROR: Unable to find a match or assign new trajectory for spot {i}, frame {self.frame}")

#     def refine_centres(self, frame, params):
#         """
#         Refines spot centers to sub-pixel precision using multiprocessing
#         with a Numba-accelerated worker function.
#         """
#         if self.num_spots == 0:
#             return

#         image_pixels = frame.as_image()
#         frame_size_tuple = tuple(frame.frame_size) # Pass frame_size as tuple
#         run_parallel_internally = (params.num_procs > 1) and (not multiprocessing.current_process().daemon)

#         # --- Create tuple of necessary parameters for Numba ---
#         # Numba's nopython mode cannot handle complex class instances like 'params' directly.
#         # Pass only the required scalar/array values as a tuple.
#         params_tuple = (
#             params.subarray_halfwidth,
#             params.inner_mask_radius,
#             params.gauss_mask_sigma,
#             params.gauss_mask_max_iter,
#             params.snr_filter_cutoff
#         )
#         # --- End parameter tuple creation ---

#         task_args = []
#         for i in range(self.num_spots):
#             # Pass the parameter tuple instead of the full params object
#             task_args.append((self.positions[i], image_pixels, frame_size_tuple, *params_tuple))


#         if run_parallel_internally:
#             # print(f"Running refine_centres internally in parallel with {params.num_procs} processes.") # Optional log
#             with contextlib.closing(multiprocessing.Pool(processes=params.num_procs)) as pool:
#                  # Use starmap if the worker function expects unpacked arguments
#                  # Here, _refine_single_spot_worker expects a single tuple 'args'
#                  # If it expected individual args: pool.starmap(_refine_single_spot_worker, task_args_unpacked)
#                  results = pool.map(_refine_single_spot_worker, task_args) # Use map for single arg tuple
#         else:
#              # print("Running refine_centres internally in serial mode.") # Optional log
#             results = []
#             for args in task_args:
#                 results.append(_refine_single_spot_worker(args))

#         # --- Unpack results and update self attributes ---
#         # (Keep the existing unpacking logic)
#         new_positions = np.zeros_like(self.positions)
#         new_bg_intensity = np.zeros_like(self.bg_intensity)
#         new_spot_intensity = np.zeros_like(self.spot_intensity)
#         new_snr = np.zeros_like(self.snr)
#         new_converged = np.zeros_like(self.converged, dtype=bool) # Use bool

#         for i, result in enumerate(results):
#             final_pos, bg_avg, spot_int, snr_val, conv_status = result
#             new_positions[i, :] = final_pos
#             new_bg_intensity[i] = bg_avg
#             new_spot_intensity[i] = spot_int
#             new_snr[i] = snr_val
#             new_converged[i] = conv_status

#         self.positions = new_positions
#         self.bg_intensity = new_bg_intensity
#         self.spot_intensity = new_spot_intensity
#         self.snr = new_snr
#         self.converged = new_converged.astype(np.int8) # Convert back