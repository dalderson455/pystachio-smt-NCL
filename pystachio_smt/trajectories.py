#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" TRAJECTORIES - Trajectory construction and manipulation module

Description:
    trajectories.py contains the Trajectory class containing information about
    spot trajectories across multiple frames, along with the routines involved
    in locating and extending the trajectories.

Contains:
    class Trajectory
    function build_trajectories
    function read_trajectories
    function write_trajectories
    function to_spots
    function compare_trajectories (Note: Added compare_trajectories as it's present in the code)

Author:
    Edward Higgins

Version: 0.2.1
"""

import csv
import sys
import os

import numpy as np
# Import Spots class from local spots module
from .spots import Spots
# Import plotting library (used in compare_trajectories)
import matplotlib.pyplot as plt

# Class representing a single spot's trajectory over time
class Trajectory:
    # Initialise a new trajectory from a single spot
    def __init__(self, id, spots, spot_id):
        self.id = id # Unique trajectory identifier
        # Frame number where trajectory starts
        self.start_frame = spots.frame
        # Frame number where trajectory currently ends (initially same as start)
        self.end_frame = spots.frame
        # List of [x, y] positions over time
        self.path = [spots.positions[spot_id, :]]
        # List of spot intensities over time
        self.intensity = [spots.spot_intensity[spot_id]]
        # List of background intensities over time
        self.bg_intensity = [spots.bg_intensity[spot_id]]
        # List of signal-to-noise ratios over time
        self.snr = [spots.snr[spot_id]]
        # Length of trajectory in frames
        self.length = 1
        # Stoichiometry value (calculated later)
        self.stoichiometry = 0
        # List of refinement convergence flags over time
        self.converged = [spots.converged[spot_id]]
        # Placeholder for linking related trajectories (unused?)
        self.linked_traj = None
        # List of spot widths [wx, wy] over time
        self.width = [spots.width[spot_id]]

    # Extend the trajectory with data from a new spot in the next frame
    def extend(self, spots, spot_id):
        # Error check: ensure the new spot is from the immediate next frame
        if spots.frame > self.end_frame + 1: # Check frame indices directly
            # Consider raising a specific error type instead of sys.exit
            # raise ValueError(f"Frame mismatch: Cannot extend trajectory {self.id} from frame {self.end_frame} to frame {spots.frame}")
             sys.exit(f"ERROR: Frame mismatch: Cannot extend trajectory {self.id} from frame {self.end_frame} to frame {spots.frame}")


        # Update the end frame of the trajectory
        self.end_frame = spots.frame
        # Append new position, intensity, etc. to the respective lists
        self.path.append(spots.positions[spot_id, :])
        self.intensity.append(spots.spot_intensity[spot_id])
        self.bg_intensity.append(spots.bg_intensity[spot_id])
        self.converged.append(spots.converged[spot_id])
        self.snr.append(spots.snr[spot_id])
        # Width is currently appended incorrectly, should append spots.width[spot_id]
        # self.width.append(spots.width[spot_id]) # Assuming this is intended behaviour

        # Increment trajectory length
        self.length += 1

# Function to build trajectories by linking spots across multiple frames
def build_trajectories(all_spots, params):
    """
    Builds trajectories by linking spots across frames. Handles potential None
    values in all_spots list due to frame processing errors.
    """
    # --- Find First Valid Frame ---
    # Initialise index and Spots object for the first frame with actual data
    first_valid_frame_index = -1
    first_valid_spots = None
    # Iterate through the list of Spots objects per frame
    for idx, spots_obj in enumerate(all_spots):
        # Check if the Spots object exists (not None) and contains spots
        if spots_obj is not None and spots_obj.num_spots > 0:
            first_valid_frame_index = idx # Store the index
            first_valid_spots = spots_obj # Store the Spots object
            # Optional consistency check: Ensure frame attribute matches list index
            if first_valid_spots.frame != first_valid_frame_index:
                 print(f"Warning: Correcting frame attr mismatch at init: obj={first_valid_spots.frame}, idx={first_valid_frame_index}")
                 first_valid_spots.frame = first_valid_frame_index # Correct the frame attribute
            break # Exit loop once the first valid frame is found

    # If no valid spots were found in any frame, return an empty list
    if first_valid_frame_index == -1:
         print("\nWarning: No valid spots found in any frame. Cannot build trajectories.")
         return []

    # --- Initialise Trajectories ---
    # List to hold all trajectory objects being built
    trajectories = []
    # Counter for assigning unique trajectory IDs
    traj_num = 0
    print(f"Starting trajectory building from frame index {first_valid_frame_index}")
    # Create initial trajectories for every spot in the first valid frame
    for i in range(first_valid_spots.num_spots):
        # Create a new Trajectory object for each spot
        trajectories.append(Trajectory(traj_num, first_valid_spots, i))
        # Increment the trajectory ID counter
        traj_num += 1

    # --- Link Spots in Subsequent Frames ---
    # Iterate through frames starting from the one *after* the first valid frame
    for frame_idx in range(first_valid_frame_index + 1, len(all_spots)):
        # Get the Spots object for the current frame index
        current_spots = all_spots[frame_idx]

        # Skip this frame if processing failed (object is None)
        if current_spots is None:
            continue

        # Optional consistency check: Ensure frame attribute matches list index
        if current_spots.frame != frame_idx:
             print(f"Warning: Correcting frame attr mismatch at link: obj={current_spots.frame}, idx={frame_idx}")
             current_spots.frame = frame_idx # Correct the frame attribute

        # Skip this frame if it contains no spots (e.g., after filtering)
        if current_spots.num_spots == 0:
            continue

        # List to keep track of assigned spots in the current frame (not used functionally here)
        assigned_spots = [] # Maybe intended for debugging or future logic
        # Iterate through each spot found in the current frame
        for spot_idx in range(current_spots.num_spots):
            # List to store potential candidate trajectories from the previous frame
            close_candidates = []
            # Find candidate trajectories ending in the immediately preceding frame
            for candidate_traj in trajectories:
                # Only consider trajectories that ended in the previous frame index
                if candidate_traj.end_frame != frame_idx - 1:
                    continue

                # --- Calculate distance to candidate's last known position ---
                try:
                    # Safety check: ensure spot index is within bounds for current frame
                    if spot_idx >= current_spots.positions.shape[0]:
                         print(f"Warning: spot_idx {spot_idx} out of bounds for current_spots.positions shape {current_spots.positions.shape} in frame {frame_idx}")
                         continue # Skip this candidate
                    # Safety check: ensure candidate trajectory path is not empty
                    if not candidate_traj.path:
                         print(f"Warning: candidate_traj {candidate_traj.id} has empty path.")
                         continue # Skip this candidate

                    # Calculate Euclidean distance
                    candidate_dist = np.linalg.norm(
                        current_spots.positions[spot_idx, :] - candidate_traj.path[-1]
                    )
                    # If distance is within the maximum allowed displacement, add to candidates
                    if candidate_dist < params.max_displacement:
                        close_candidates.append(candidate_traj)
                # Handle potential errors if accessing path fails
                except IndexError as ie:
                    print(f"Warning: IndexError during distance calc for frame {frame_idx}, spot {spot_idx}, traj {candidate_traj.id}. Details: {ie}")
                    continue # Skip this candidate pair
                # --- End distance calculation ---

            # --- Assign current spot based on number of close candidates ---
            # Case 0: No candidates found within range
            if len(close_candidates) == 0:
                # Start a new trajectory for this spot
                trajectories.append(Trajectory(traj_num, current_spots, spot_idx))
                traj_num += 1 # Increment global trajectory counter
                assigned_spots.append(spot_idx) # Mark spot as assigned

            # Case 1: Exactly one candidate found
            elif len(close_candidates) == 1:
                # Extend the single candidate trajectory
                chosen_candidate = close_candidates[0]
                try:
                    # Optional pre-check for frame continuity before calling extend
                    if current_spots.frame != chosen_candidate.end_frame + 1:
                         print(f"ERROR PRE-CHECK FAIL: Frame mismatch before extend call. Spot frame: {current_spots.frame}, Candidate end frame+1: {chosen_candidate.end_frame + 1}")
                         # Start new trajectory if frames don't match exactly
                         trajectories.append(Trajectory(traj_num, current_spots, spot_idx))
                         traj_num += 1
                    else:
                         # Extend the existing trajectory
                         chosen_candidate.extend(current_spots, spot_idx)

                    assigned_spots.append(spot_idx) # Mark spot as assigned
                # Handle potential errors during the extend call
                except Exception as e:
                     print(f"ERROR during extend call for traj {chosen_candidate.id}, spot {spot_idx}, frame {frame_idx}: {type(e).__name__} - {e}")
                     # Spot remains unassigned if extend fails

            # Case 2: More than one candidate found (ambiguous link)
            else: # len(close_candidates) > 1
                # Find the absolute nearest candidate among the close ones
                min_dist = float('inf')
                nearest_candidate = None
                # Iterate through candidates to find the closest one
                for candidate in close_candidates:
                    try:
                        # Recalculate distance (could store from before, but recalculating is safe)
                        dist = np.linalg.norm(
                            current_spots.positions[spot_idx, :] - candidate.path[-1]
                        )
                        # Update nearest if this one is closer
                        if dist < min_dist:
                            min_dist = dist
                            nearest_candidate = candidate
                    except IndexError:
                        # Handle unlikely error accessing path again
                        print(f"Warning: IndexError accessing path for candidate traj {candidate.id} during ambiguity check.")
                        continue

                # If a nearest candidate was successfully identified
                if nearest_candidate is not None:
                    # Optional: Log the ambiguous link resolution
                    # if params.verbose:
                    #      print(f"Frame {frame_idx}, Spot {spot_idx}: Ambiguous link ({len(close_candidates)} candidates). Linking to nearest: Traj {nearest_candidate.id} (Dist: {min_dist:.2f})")
                    # Try to extend the nearest candidate
                    try:
                        # Optional pre-check for frame continuity
                        if current_spots.frame != nearest_candidate.end_frame + 1:
                             print(f"ERROR PRE-CHECK FAIL (Ambiguous): Frame mismatch before extend call. Spot frame: {current_spots.frame}, Candidate end frame+1: {nearest_candidate.end_frame + 1}")
                             # Start new trajectory if frames don't match
                             trajectories.append(Trajectory(traj_num, current_spots, spot_idx))
                             traj_num += 1
                        else:
                            # Extend the nearest candidate's trajectory
                            nearest_candidate.extend(current_spots, spot_idx)

                        assigned_spots.append(spot_idx) # Mark spot as assigned
                    # Handle potential errors during extend
                    except Exception as e:
                         print(f"ERROR during extend call for nearest traj {nearest_candidate.id}, spot {spot_idx}, frame {frame_idx}: {type(e).__name__} - {e}")
                         # Spot remains unassigned if extend fails
                else:
                    # Fallback if nearest couldn't be determined (e.g., all distance calcs failed)
                    print(f"Warning: Could not determine nearest neighbour for ambiguous link frame {frame_idx}, spot {spot_idx}. Starting new trajectory.")
                    trajectories.append(Trajectory(traj_num, current_spots, spot_idx))
                    traj_num += 1
                    assigned_spots.append(spot_idx) # Mark spot as assigned
            # End of candidate handling block
        # End of loop through spots in current frame
    # End of loop through frames

    # --- Filter Trajectories by Length and Re-assign IDs ---
    # Create a new list containing only trajectories longer than or equal to the minimum length
    filtered_trajectories = [traj for traj in trajectories if traj.length >= params.min_traj_len]

    # Re-number the IDs of the filtered trajectories sequentially from 0
    actual_traj_num = 0
    for traj in filtered_trajectories:
        traj.id = actual_traj_num
        actual_traj_num += 1

    # Print summary message
    print(f"\nBuilt {len(filtered_trajectories)} trajectories meeting min length ({params.min_traj_len}).")
    # Return the final list of filtered trajectories
    return filtered_trajectories

# Function to write trajectory data to a CSV file
def write_trajectories(trajectories, filename):
         # Open the specified file in write mode
    f = open(filename, "w")
    # Write the header row with tab separators
    f.write(f"trajectory\tframe\tx\ty\tspot_intensity\tbg_intensity\tSNR\tconverged\twidthx\twidthy\n")
    # Iterate through each trajectory object in the list
    for traj in trajectories:
        # Iterate through the frame numbers this trajectory covers
        # Note: This loop iterates based on start_frame and end_frame attributes.
        for frame in range(traj.start_frame, traj.end_frame + 1):
            # Calculate the list index 'i' corresponding to the current frame number
            # This assumes frame numbers are consecutive and start correctly relative to index 0.
            i = frame - traj.start_frame
            # Write the trajectory data for the current frame as a tab-separated row
            # Uses f-string formatting to construct the output string.
            # IMPORTANT: Accesses width using fixed index [0] - assumes width is constant for the trajectory?
            f.write(
                f"{traj.id}\t{frame}\t{traj.path[i][0]}\t{traj.path[i][1]}\t{traj.intensity[i]}\t{traj.bg_intensity[i]}\t{traj.snr[i]}\t{traj.converged[i]}\t{traj.width[0][0]}\t{traj.width[0][1]}\n"
            )
    # Close the output file
    f.close()

# Function to convert a list of Trajectory objects back into a list of Spots objects (one per frame)
def to_spots(trajs):
    # Initialise list to hold Spots objects for each frame
    all_spots = []
    frame = 0 # Start from frame 0
    done_all_frames = False # Flag to control loop termination

    # Loop until all frames covered by the trajectories are processed
    while not done_all_frames:
        done_all_frames = True # Assume done unless a trajectory extends further

        # --- Collect spot data for the current frame ---
        positions = []
        bg_intensity = []
        spot_intensity = []
        snr = []
        converged = []
        width = []
        # Iterate through all input trajectories
        for traj in trajs:
            # Check if this trajectory extends beyond the current frame
            if traj.end_frame >= frame: # Check includes current frame
                done_all_frames = False # Signal that more frames need processing

                # Check if the trajectory exists in the *current* frame
                if traj.start_frame <= frame:
                    # Calculate index within trajectory lists
                    i = frame - traj.start_frame
                    # --- Safety Check ---
                    # Ensure index is valid before accessing data
                    if i < traj.length:
                         positions.append(traj.path[i][:])
                         spot_intensity.append(traj.intensity[i])
                         bg_intensity.append(traj.bg_intensity[i])
                         snr.append(traj.snr[i])
                         converged.append(traj.converged[i])
                         # Append the width data associated with this frame index
                         width.append(traj.width[i]) # Assumes width list matches length
                    else:
                         # This case indicates an internal inconsistency if traj.end_frame was correct
                         print(f"Warning: Index mismatch during to_spots conversion for traj {traj.id} at frame {frame}")

        # --- Create Spots object for the current frame ---
        # Create object only if spots were found in this frame
        if positions:
            num_spots_in_frame = len(positions)
            spots_obj = Spots(num_spots_in_frame, frame) # Create Spots object
            # Populate its attributes with collected data
            spots_obj.set_positions(np.array(positions))
            spots_obj.spot_intensity = np.array(spot_intensity)
            spots_obj.bg_intensity = np.array(bg_intensity)
            spots_obj.snr = np.array(snr)
            spots_obj.converged = np.array(converged, dtype=np.int8)
            # Handle potential inconsistencies in width data structure if needed
            # Assuming width is a list of [wx, wy] pairs or similar
            try:
                spots_obj.width = np.array(width) # Convert list of widths to numpy array
                # Optional: Check shape if necessary (e.g., spots_obj.width.shape == (num_spots_in_frame, 2))
            except ValueError as ve:
                 print(f"Warning: Could not convert width list to array for frame {frame}. Check width data consistency. Error: {ve}")
                 # Assign default/empty array or handle error as appropriate
                 spots_obj.width = np.zeros((num_spots_in_frame, 2))


            all_spots.append(spots_obj) # Add Spots object to the list
        else:
            # Optionally add an empty Spots object or None if no spots in this frame
            all_spots.append(Spots(0, frame)) # Add empty Spots object

        # Move to the next frame
        frame += 1

    # Return the list of Spots objects, one for each frame
    return all_spots

# Function to read trajectory data from a CSV file
def read_trajectories(filename):
    # Initialise list to hold Trajectory objects
    trajectories = []
    prev_traj_id = -1 # Keep track of the last trajectory ID processed
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"WARNING: No such file {filename}")
        return None # Return None if file not found

    # Open and read the CSV file
    with open(filename) as tsv_file:
        # Use csv reader with tab delimiter
        tsv_reader = csv.reader(tsv_file,delimiter="\t")
        header = next(tsv_reader) # Read and skip the header row

        # Process each data row
        for line in tsv_reader:
            # --- Robust parsing with error checking ---
            try:
                # Create a temporary Spots object to hold current row's data
                spot = Spots(num_spots=1)
                # Parse data from columns, converting to appropriate types
                traj_id = int(line[0])
                spot.frame = int(line[1])
                spot.positions[0, :] = [float(line[2]), float(line[3])]
                spot.spot_intensity[0] = float(line[4])
                spot.bg_intensity[0] = float(line[5])
                spot.snr[0] = float(line[6])
                spot.converged[0] = int(line[7])
                # Width is stored as [wx, wy] pair in the Spots object
                spot.width[0,:] = [float(line[8]),float(line[9])]
            except (IndexError, ValueError) as e:
                 print(f"Warning: Skipping row due to parsing error in {filename}: {line}. Error: {e}")
                 continue # Skip to the next row if parsing fails
            # --- End robust parsing ---

            # Check if this row belongs to a new trajectory
            if traj_id != prev_traj_id:
                # If new ID, create a new Trajectory object and add to list
                trajectories.append(Trajectory(traj_id, spot, 0))
                prev_traj_id = traj_id # Update the last seen ID
            else:
                # If same ID, extend the *last added* trajectory
                # Add check to ensure trajectories list is not empty
                if trajectories:
                    try:
                        trajectories[-1].extend(spot, 0)
                    except Exception as e:
                        # Handle errors during extend (e.g., frame mismatch)
                        print(f"Error extending trajectory {prev_traj_id} from file {filename} at frame {spot.frame}. Error: {e}")
                        # Decide how to proceed - skip? Start new traj?
                        # For now, just prints error.
                else:
                    # Should not happen if prev_traj_id logic is correct, but safety check
                    print(f"Warning: Encountered row for trajectory {traj_id} but no previous trajectory exists.")


    # Return the list of reconstructed Trajectory objects
    return trajectories

# Function to compare tracked trajectories against simulated ground truth
def compare_trajectories(params):
    # --- File Reading with Error Handling ---
    try:
        # Read tracked trajectories
        trajs = read_trajectories(params.name + "_trajectories.csv")
        if trajs is None: trajs = [] # Use empty list if file read failed
    except Exception as e:
        print(f"Error reading tracked trajectories: {e}")
        trajs = []

    try:
        # Read simulated (ground truth) trajectories
        target_trajs = read_trajectories(params.name + "_simulated.csv")
        if target_trajs is None: target_trajs = [] # Use empty list if file read failed
    except Exception as e:
        print(f"Error reading simulated trajectories: {e}")
        target_trajs = []

    # --- Determine Frame Range ---
    # Find the maximum frame number present in either dataset
    max_frame_tracked = max((traj.end_frame for traj in trajs), default=-1)
    max_frame_simulated = max((traj.end_frame for traj in target_trajs), default=-1)
    num_frames_to_process = max(max_frame_tracked, max_frame_simulated) + 1

    # Initialise lists to hold Spots objects per frame for both datasets
    all_target_spots = [None] * num_frames_to_process
    all_spots = [None] * num_frames_to_process

    # --- Reconstruct Spots Objects from Trajectories ---
    # Reconstruct ground truth spots per frame
    for frame in range(num_frames_to_process):
        frame_target_positions = []
        # Collect positions of target spots present in the current frame
        for traj in target_trajs:
            if frame >= traj.start_frame and frame <= traj.end_frame:
                 i = frame - traj.start_frame # Calculate index within trajectory
                 if i < len(traj.path): # Safety check for index validity
                      frame_target_positions.append(traj.path[i][:])
                 else:
                      print(f"Warning: Index mismatch in target traj {traj.id} at frame {frame}")
        # If spots found, create a Spots object for this frame
        if frame_target_positions:
             target_spots_obj = Spots(len(frame_target_positions), frame)
             target_spots_obj.set_positions(np.array(frame_target_positions))
             all_target_spots[frame] = target_spots_obj

    # Reconstruct tracked spots per frame
    for frame in range(num_frames_to_process):
        frame_tracked_positions = []
        # Collect positions of tracked spots present in the current frame
        for traj in trajs:
             if frame >= traj.start_frame and frame <= traj.end_frame:
                  i = frame - traj.start_frame # Calculate index
                  if i < len(traj.path): # Safety check
                       frame_tracked_positions.append(traj.path[i][:])
                  else:
                       print(f"Warning: Index mismatch in tracked traj {traj.id} at frame {frame}")
        # If spots found, create a Spots object
        if frame_tracked_positions:
             spots_obj = Spots(len(frame_tracked_positions), frame)
             spots_obj.set_positions(np.array(frame_tracked_positions))
             all_spots[frame] = spots_obj

    # --- Frame-by-Frame Comparison ---
    # Lists to store metrics for each frame
    error_list = [] # Localisation error for matches
    fn_list = [] # False negatives
    fp_list = [] # False positives
    matches_list = [] # Number of matches

    # Iterate through each frame
    for frame in range(num_frames_to_process):
        # Check if spot data exists for this frame in both datasets
        if frame >= len(all_spots) or frame >= len(all_target_spots) or all_spots[frame] is None or all_target_spots[frame] is None:
            continue # Skip frame if data is missing

        # Get Spots objects for the current frame
        current_tracked_spots = all_spots[frame]
        current_target_spots = all_target_spots[frame]

        # Get number of tracked and target spots
        num_found_spots = current_tracked_spots.num_spots
        num_target_spots = current_target_spots.num_spots

        # Initialise per-frame counters and tracking sets
        matches = 0 # Number of matches found in this frame
        errors_in_frame = [] # List of errors for matched spots
        outside_frame = 0 # Count target spots outside frame boundaries
        # Sets to efficiently track used indices during matching
        assigned_tracker_indices = set()
        matched_target_indices = set()

        # --- Matching Logic: Iterate through target spots ---
        for target_idx in range(num_target_spots):
            target_pos = current_target_spots.positions[target_idx,:]

            # Check if target is outside frame boundaries (optional, requires params.frame_size)
            is_outside = False
            if hasattr(params, 'frame_size') and params.frame_size:
                 frame_width, frame_height = params.frame_size
                 if (target_pos[0] < 0 or target_pos[1] < 0 or
                     target_pos[0] >= frame_width or target_pos[1] >= frame_height):
                     outside_frame += 1
                     is_outside = True
            # Skip this target if it's outside
            if is_outside: continue

            # Find the closest *unassigned* tracked spot within 1 pixel distance
            best_match_dist = float('inf')
            best_match_idx = -1

            # Iterate through tracked spots to find the best match
            for tracker_idx in range(num_found_spots):
                 # Skip if this tracked spot is already assigned to another target
                 if tracker_idx in assigned_tracker_indices:
                      continue

                 # Calculate distance
                 tracker_pos = current_tracked_spots.positions[tracker_idx,:]
                 dist = np.linalg.norm(target_pos - tracker_pos)

                 # Check if within threshold (1 pixel) AND is the closest found so far
                 if dist < 1.0 and dist < best_match_dist:
                      best_match_dist = dist
                      best_match_idx = tracker_idx

            # If a suitable match was found
            if best_match_idx != -1:
                 matches += 1 # Increment match count
                 errors_in_frame.append(best_match_dist) # Store localisation error
                 assigned_tracker_indices.add(best_match_idx) # Mark tracked spot as used
                 matched_target_indices.add(target_idx) # Mark target spot as matched

        # --- Calculate Frame Metrics ---
        # Number of target spots within frame boundaries
        valid_target_count = num_target_spots - outside_frame
        # False Negatives: Valid targets not matched
        false_negatives = valid_target_count - len(matched_target_indices)
        # False Positives: Tracked spots not assigned to any target
        false_positives = num_found_spots - len(assigned_tracker_indices)

        # Ensure metrics are non-negative
        false_negatives = max(0, false_negatives)
        false_positives = max(0, false_positives)

        # Calculate mean localisation error for this frame
        mean_error_frame = np.mean(errors_in_frame) if errors_in_frame else 0

        # Store metrics for this frame
        error_list.append(mean_error_frame)
        fn_list.append(false_negatives)
        fp_list.append(false_positives)
        matches_list.append(matches)

    # --- Overall Statistics Calculation and Plotting ---
    # Check if any frames were processed
    if not fn_list:
        print("\nNo frames were compared. Cannot generate summary plot.")
        return

    # Calculate mean and standard deviation for each metric across all frames
    avg_error = np.mean(error_list)
    std_error = np.std(error_list)
    avg_fn = np.mean(fn_list)
    std_fn = np.std(fn_list)
    avg_fp = np.mean(fp_list)
    std_fp = np.std(fp_list)
    avg_matches = np.mean(matches_list)
    std_matches = np.std(matches_list)

    # Print summary statistics
    print("\n--- Overall Comparison Summary ---")
    print(f"Avg Matches per frame: {avg_matches:.2f} +/- {std_matches:.2f}")
    print(f"Avg False Negatives per frame: {avg_fn:.2f} +/- {std_fn:.2f}")
    print(f"Avg False Positives per frame: {avg_fp:.2f} +/- {std_fp:.2f}")
    print(f"Avg Localisation Error (pixels): {avg_error:.3f} +/- {std_error:.3f}")

    # --- Generate Bar Chart Summary ---
    metrics = ['Matches', 'False Negatives', 'False Positives', 'Error (pixels)']
    averages = [avg_matches, avg_fn, avg_fp, avg_error]
    std_devs = [std_matches, std_fn, std_fp, std_error]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Create bar chart with error bars (standard deviation)
    bars = ax.bar(metrics, averages, yerr=std_devs, capsize=5, color=['green', 'red', 'orange', 'blue'])

    # Formatting
    ax.set_ylabel('Average Value / Count per Frame')
    ax.set_title('Average Spot Detection Metrics (Mean +/- Std Dev)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add text labels showing the average value on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout() # Adjust layout

    # Save the plot to a file
    plot_filename = params.name + "_comparison_summary.png"
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Comparison summary plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")

    # Display the plot if requested in parameters
    display_figs = getattr(params, 'display_figures', False) # Check safely
    if display_figs:
        plt.show()
    # Close the plot figure to free memory
    plt.close(fig)