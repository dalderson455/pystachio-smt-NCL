# #! /usr/bin/env python3
# # -*- coding: utf-8 -*-
# # vim:fenc=utf-8
# #
# # Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
# #
# # Distributed under terms of the MIT license.

# """ TRACKING - Spot tracking module

# Description:
#     tracking.py contains the code for the tracking task, which identifies spots
#     within a set of frames and builds spot trajectories across those frames.

# Contains:
#     function track
#     function track_frame

# Author:
#     Edward Higgins

# Version: 0.2.1
# """

""" TRACKING - Spot tracking module

Description:
    tracking.py contains the code for the tracking task, which identifies spots
    within a set of frames and builds spot trajectories across those frames.

Contains:
    function track
    function track_frame

Author:
    Edward Higgins

Version: 0.2.1
"""

# --- Core library imports ---
import sys
import numpy as np
import multiprocessing as mp
import contextlib
import traceback

# --- Local module imports ---
from . import spots
from . import trajectories
from . import images

# --- Main tracking function ---
def track(params):
    # Initialise image data container
    image_data = images.ImageData()
    # Load image stack from file specified in parameters
    image_data.read(params.name + ".tif", params)

    # --- Handle Alternating Laser EXcitation (ALEX) ---
    if params.ALEX==True:
         # Placeholder for ALEX-specific processing logic
         pass

    # --- Handle standard (non-ALEX) tracking ---
    else:
        # List to store Spots objects from each processed frame
        all_spots = []
        # Aggregate counters for Gaussian fit statistics
        overall_success_count = 0
        overall_total_count = 0

        # --- SERIAL execution path ---
        if params.num_procs <= 1:
            # Loop through each frame index
            for frame in range(image_data.num_frames):
                try:
                    # Process frame, get spots object and fit stats
                    frame_spots_result, success_count, total_count = track_frame(image_data[frame], frame, params)
                    # If processing succeeded, store results
                    if frame_spots_result is not None:
                        all_spots.append(frame_spots_result)
                        overall_success_count += success_count
                        overall_total_count += total_count
                    # Log skipped frame if failed and verbose
                    elif params.verbose:
                         print(f"Skipping aggregation for failed frame {frame}.")
                # Catch unexpected errors during serial processing
                except Exception as e:
                     print(f"Critical error during serial processing of frame {frame}: {type(e).__name__} - {e}")

        # --- PARALLEL execution path ---
        else:
            # List to hold async results from worker pool
            res = [None] * image_data.num_frames
            # Create and manage multiprocessing pool safely
            with contextlib.closing(mp.Pool(processes=params.num_procs)) as pool:
                # Submit all frames for processing asynchronously
                for frame in range(image_data.num_frames):
                    res[frame] = pool.apply_async(track_frame, (image_data[frame], frame, params))

                # Retrieve results as they become available
                for frame in range(image_data.num_frames):
                    try:
                        # Get result tuple from async object (blocks if not ready)
                        result = res[frame].get()
                        frame_spots_result, success_count, total_count = result

                        # If processing succeeded, store results
                        if frame_spots_result is not None:
                            all_spots.append(frame_spots_result)
                            overall_success_count += success_count
                            overall_total_count += total_count
                        # Log skipped frame if failed and verbose
                        elif params.verbose:
                             print(f"Skipping aggregation for failed frame {frame}.")
                    # Catch errors during result retrieval/unpacking
                    except Exception as e:
                        print(f"Error retrieving/unpacking result for frame {frame}: {type(e).__name__} - {e}")

        # --- Post-tracking processing ---
        # Report overall Gaussian fit statistics if verbose
        if params.verbose:
             if overall_total_count > 0:
                 overall_rate = (overall_success_count / overall_total_count) * 100
                 print(f"\n--- Overall Gaussian Fit Summary ---")
                 print(f"Total Fits Attempted: {overall_total_count}")
                 print(f"Total Successful Fits: {overall_success_count}")
                 print(f"Overall Success Rate: {overall_rate:.1f}%")
             else:
                 print(f"\n--- Overall Gaussian Fit Summary ---")
                 print(f"No spots processed for width fitting.")

        # Build trajectories from collected Spots objects
        trajs = trajectories.build_trajectories(all_spots, params)
        # Write trajectories to CSV file if created
        if trajs:
             trajectories.write_trajectories(trajs, params.name + "_trajectories.csv")
        # Report if no trajectories were built
        else:
             print("\nNo trajectories built.")


# --- Single frame processing function (called serially or by workers) ---
def track_frame(frame_data, frame, params):
    # Initialise frame-specific counters for Gaussian fits
    num_success_frame = 0
    total_processed_frame = 0
    # Initialise a Spots object for the frame
    frame_spots = spots.Spots(frame=frame)
    try:
        # --- Spot Detection & Characterisation Pipeline ---
        # 1. Find initial candidates
        sys.stdout.flush() # Ensure prior prints are flushed
        frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
        found_spots = frame_spots.num_spots

        # 2. Merge close candidates
        sys.stdout.flush()
        frame_spots.merge_coincident_candidates()
        merged_spots = frame_spots.num_spots

        # 3. Refine centres (if spots remain)
        sys.stdout.flush()
        if frame_spots.num_spots > 0:
             frame_spots.refine_centres(frame_data, params)
             # Check for NaN/inf values after refinement
             if np.any(np.isnan(frame_spots.positions)) or np.any(np.isinf(frame_spots.positions)):
                  print(f"!!! WARNING: Frame {frame} - NaN/inf detected in positions after refine_centres.")
             if np.any(np.isnan(frame_spots.snr)) or np.any(np.isinf(frame_spots.snr)):
                  print(f"!!! WARNING: Frame {frame} - NaN/inf detected in SNR after refine_centres.")
             if np.any(np.isnan(frame_spots.converged)):
                  print(f"!!! WARNING: Frame {frame} - NaN detected in converged after refine_centres.")
             sys.stdout.flush()
        else:
             sys.stdout.flush()

        # 4. Filter spots (if spots remain)
        sys.stdout.flush()
        if frame_spots.num_spots > 0:
             frame_spots.filter_candidates(frame_data, params)
        else:
             sys.stdout.flush()

        # 5. Calculate intensities (if spots remain)
        sys.stdout.flush()
        if frame_spots.num_spots > 0:
             frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
        else:
             sys.stdout.flush()

        # 6. Calculate widths (if spots remain)
        sys.stdout.flush()
        if frame_spots.num_spots > 0:
             num_success_frame, total_processed_frame = frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
        else:
             # Ensure counters are zero if no spots processed for width
             sys.stdout.flush()
             num_success_frame, total_processed_frame = 0, 0

        # --- Pipeline End ---

        # Print verbose frame summary if enabled
        if params.verbose:
            # Note: Original summary print statement is commented out below for reference
            # print(
            #     f"Frame {frame}: Final {frame_spots.num_spots:3d} spots "
            #     f"({found_spots:3d} identified, "
            #     f"{found_spots-merged_spots:3d} merged, "
            #     f"{merged_spots-frame_spots.num_spots:3d} filtered)"
            # )
            sys.stdout.flush()

        # Final flush before returning results
        sys.stdout.flush()

        # Return processed spots object and frame's fit statistics
        return frame_spots, num_success_frame, total_processed_frame

    # Catch any exceptions during the pipeline
    except Exception as e:
        # Print simplified error information
        print(f"!!! EXCEPTION CAUGHT in track_frame for frame {frame} !!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Args: {e.args}")
        # Attempt to print traceback for detailed debugging
        try:
             print("--- Traceback ---")
             traceback.print_exc()
             print("--- End Traceback ---")
        except Exception as e_tb:
             print(f"ERROR printing traceback: {type(e_tb)} - {e_tb.args}")
        sys.stdout.flush() # Ensure error output is flushed
        # Return None to signal failure for this frame
        return None, 0, 0

# import sys

# import numpy as np
# import multiprocessing as mp
# #import matplotlib.pyplot as plt
# # Import local modules
# from . import spots
# from . import trajectories
# from . import images
# # Import contextlib for safe multiprocessing pool management
# import contextlib
# # Import traceback for detailed error logging (optional)
# import traceback

# # Main function to perform tracking based on parameters
# def track(params):
#     # Create ImageData object to hold image stack
#     image_data = images.ImageData()
#     # Read image file (e.g., .tif) based on name in params
#     image_data.read(params.name + ".tif", params)

#     # --- Handle Alternating Laser EXcitation (ALEX) experiments ---
#     if params.ALEX==True:
#          # (Existing ALEX processing logic - comments omitted for brevity)
#          # Need to handle splitting channels, tracking each, and potentially aggregating stats
#          pass # Placeholder for ALEX logic

#     # --- Handle standard (non-ALEX) experiments ---
#     else:
#         # List to store Spots objects for each processed frame
#         all_spots = []
#         # Initialise counters for overall Gaussian fit statistics
#         overall_success_count = 0
#         overall_total_count = 0

#         # --- SERIAL EXECUTION (if num_procs <= 1) ---
#         if params.num_procs <= 1:
#             # Process each frame sequentially
#             for frame in range(image_data.num_frames):
#                 try:
#                     # Process the frame, get Spots object and fit stats
#                     frame_spots_result, success_count, total_count = track_frame(image_data[frame], frame, params)
#                     # If processing succeeded (returned Spots object)
#                     if frame_spots_result is not None:
#                         all_spots.append(frame_spots_result) # Add to list
#                         # Aggregate fit statistics
#                         overall_success_count += success_count
#                         overall_total_count += total_count
#                     else:
#                         # Log skipped frame if verbose
#                         if params.verbose:
#                              print(f"Skipping aggregation for failed frame {frame}.")
#                 # Catch any unexpected errors during serial processing
#                 except Exception as e:
#                      print(f"Critical error during serial processing of frame {frame}: {type(e).__name__} - {e}")
#                      # Currently skips frame on critical error

#         # --- PARALLEL EXECUTION (if num_procs > 1) ---
#         else:
#             # Array to hold asynchronous results from workers
#             res = [None] * image_data.num_frames
#             # Use contextlib.closing for safe Pool management (ensures closure)
#             with contextlib.closing(mp.Pool(processes=params.num_procs)) as pool:
#                 # Submit all frame processing jobs to the pool
#                 for frame in range(image_data.num_frames):
#                     res[frame] = pool.apply_async(track_frame, (image_data[frame], frame, params))

#                 # Retrieve and process results from workers
#                 for frame in range(image_data.num_frames):
#                     try:
#                         # Get result tuple (may block until worker finishes)
#                         result = res[frame].get()
#                         # Unpack the result: Spots object, fit success count, fit total count
#                         frame_spots_result, success_count, total_count = result

#                         # If worker returned a valid Spots object
#                         if frame_spots_result is not None:
#                             all_spots.append(frame_spots_result) # Add to list
#                             # Aggregate fit statistics
#                             overall_success_count += success_count
#                             overall_total_count += total_count
#                         else:
#                             # Log skipped frame if verbose
#                             if params.verbose:
#                                  print(f"Skipping aggregation for failed frame {frame}.")
#                     # Catch errors during result retrieval or unpacking
#                     except Exception as e:
#                         print(f"Error retrieving/unpacking result for frame {frame}: {type(e).__name__} - {e}")
#                         # Continues to next frame's result

#             # Pool is automatically closed here

#         # --- Post-Processing and Output ---
#         # Print overall Gaussian fit statistics if verbose
#         if params.verbose:
#              if overall_total_count > 0:
#                  overall_rate = (overall_success_count / overall_total_count) * 100
#                  print(f"\n--- Overall Gaussian Fit Summary ---")
#                  print(f"Total Fits Attempted: {overall_total_count}")
#                  print(f"Total Successful Fits: {overall_success_count}")
#                  print(f"Overall Success Rate: {overall_rate:.1f}%")
#              else:
#                  print(f"\n--- Overall Gaussian Fit Summary ---")
#                  print(f"No spots processed for width fitting.")

#         # Build trajectories from the list of detected spots per frame
#         trajs = trajectories.build_trajectories(all_spots, params)
#         # Write trajectories to a CSV file if any were built
#         if trajs:
#              trajectories.write_trajectories(trajs, params.name + "_trajectories.csv") # Use CSV extension
#         else:
#              # Inform user if no trajectories were generated
#              print("\nNo trajectories built.")

# # Function to process a single frame: find, refine, characterise spots
# # def track_frame(frame_data, frame, params):
# #     # Initialise frame-specific counters for Gaussian fit success
# #     num_success_frame = 0
# #     total_processed_frame = 0
# #     # Initialise Spots object (ensures it exists even if errors occur)
# #     frame_spots = spots.Spots(frame=frame)
# #     try:
# #         # --- Spot Detection and Characterisation Pipeline ---
# #         # 1. Find initial spot candidates
# #         frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
# #         found_spots = frame_spots.num_spots # Store initial count
# #         # 2. Merge spots that are too close
# #         frame_spots.merge_coincident_candidates()
# #         merged_spots = frame_spots.num_spots # Store count after merging
# #         # 3. Refine spot centres to sub-pixel accuracy
# #         frame_spots.refine_centres(frame_data, params)
# #         # 4. Filter spots based on SNR, edge proximity, mask
# #         frame_spots.filter_candidates(frame_data, params)
# #         # 5. Calculate background-corrected intensities
# #         frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
# #         # 6. Calculate spot widths using Gaussian fitting, get fit stats
# #         num_success_frame, total_processed_frame = frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
# #         # --- End Pipeline ---

# #         # Print frame summary if verbose mode is on
# #         if params.verbose:
# #             print(
# #                 f"Frame {frame:4d}: found {frame_spots.num_spots:3d} spots "
# #                 f"({found_spots:3d} identified, "
# #                 f"{found_spots-merged_spots:3d} merged, "
# #                 f"{merged_spots-frame_spots.num_spots:3d} filtered)"
# #             )
# #         # Return the processed Spots object and fit stats on success
# #         return frame_spots, num_success_frame, total_processed_frame

# #     # Catch any exception during frame processing
# #     except Exception as e:
# #         # Print error message indicating the frame and error type
# #         print(f"!!! ERROR processing frame {frame}: {type(e).__name__} - {e}")
# #         # Optional: Print full traceback for detailed debugging
# #         # traceback.print_exc()
# #         # Return None and zero counts to signal failure to the main process
# #         return None, 0, 0
# def track_frame(frame_data, frame, params):
#     # Initialise frame-specific counters for Gaussian fit success
#     num_success_frame = 0
#     total_processed_frame = 0
#     # Initialise Spots object (ensures it exists even if errors occur)
#     frame_spots = spots.Spots(frame=frame)
#     try:
#         # --- Spot Detection and Characterisation Pipeline ---
#         # (Keep the steps: find_in_frame, merge, refine, filter, get_intensity, get_widths)
#         # ... (existing code for these steps) ...

#         #print(f"Frame {frame}: Starting find_in_frame...")
#         sys.stdout.flush() # Force print output immediately
#         frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
#         found_spots = frame_spots.num_spots

#         #print(f"Frame {frame}: Starting merge_coincident_candidates... (found {found_spots})")
#         sys.stdout.flush()
#         frame_spots.merge_coincident_candidates()
#         merged_spots = frame_spots.num_spots

#         #print(f"Frame {frame}: Starting refine_centres... (merged {frame_spots.num_spots})")
#         sys.stdout.flush()
#         if frame_spots.num_spots > 0:
#              frame_spots.refine_centres(frame_data, params)
#              # ***** ADD NaN/inf CHECKS *****
#              if np.any(np.isnan(frame_spots.positions)) or np.any(np.isinf(frame_spots.positions)):
#                   print(f"!!! WARNING: Frame {frame} - NaN/inf detected in positions after refine_centres.")
#              if np.any(np.isnan(frame_spots.snr)) or np.any(np.isinf(frame_spots.snr)):
#                   print(f"!!! WARNING: Frame {frame} - NaN/inf detected in SNR after refine_centres.")
#              # Check convergence array too, though less likely to be NaN/inf
#              if np.any(np.isnan(frame_spots.converged)):
#                   print(f"!!! WARNING: Frame {frame} - NaN detected in converged after refine_centres.")
#              sys.stdout.flush()
#         else:
#              #print(f"Frame {frame}: Skipping refine_centres (no spots after merge).")
#              sys.stdout.flush()

#         #print(f"Frame {frame}: Starting filter_candidates...")
#         sys.stdout.flush()
#         if frame_spots.num_spots > 0:
#              frame_spots.filter_candidates(frame_data, params)
#         else:
#              #print(f"Frame {frame}: Skipping filter_candidates (no spots after refine).")
#              sys.stdout.flush()

#         #print(f"Frame {frame}: Starting get_spot_intensities... (filtered {frame_spots.num_spots})")
#         sys.stdout.flush()
#         if frame_spots.num_spots > 0:
#              frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
#         else:
#              #print(f"Frame {frame}: Skipping get_spot_intensities (no spots after filter).")
#              sys.stdout.flush()

#         #print(f"Frame {frame}: Starting get_spot_widths...")
#         sys.stdout.flush()
#         if frame_spots.num_spots > 0:
#              num_success_frame, total_processed_frame = frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
#         else:
#              #print(f"Frame {frame}: Skipping get_spot_widths (no spots remaining).")
#              sys.stdout.flush()
#              num_success_frame, total_processed_frame = 0, 0 # Ensure defined

#         # Print frame summary if verbose mode is on
#         if params.verbose:
#             # print(
#             #     f"Frame {frame}: Final {frame_spots.num_spots:3d} spots "
#             #     f"({found_spots:3d} identified, "
#             #     f"{found_spots-merged_spots:3d} merged, "
#             #     f"{merged_spots-frame_spots.num_spots:3d} filtered)"
#             # )
#             sys.stdout.flush()

#         # ***** ADD PRINT BEFORE RETURN *****
#         #print(f"Frame {frame}: Attempting to return: spots_obj type={type(frame_spots)}, success={num_success_frame}, total={total_processed_frame}")
#         sys.stdout.flush()
#         # ***** END ADD PRINT *****

#         # Return the processed Spots object and fit stats on success
#         return frame_spots, num_success_frame, total_processed_frame

#     # Catch any exception during frame processing
#     except Exception as e:
#         # ***** SIMPLIFY EXCEPTION PRINTING *****
#         print(f"!!! EXCEPTION CAUGHT in track_frame for frame {frame} !!!")
#         # Try printing just the exception type and args
#         print(f"Exception Type: {type(e)}")
#         print(f"Exception Args: {e.args}")
#         # Try traceback separately
#         try:
#              print("--- Traceback ---")
#              traceback.print_exc()
#              print("--- End Traceback ---")
#         except Exception as e_tb:
#              print(f"ERROR printing traceback: {type(e_tb)} - {e_tb.args}")
#         sys.stdout.flush() # Ensure everything is printed
#         # ***** END SIMPLIFY *****
#         # DO NOT re-raise for now, let it return None
#         # raise e
#         return None, 0, 0