# Understanding `tracking.py`: Single Molecule Tracking Analysis

The tracking.py module is a core component of the SMT framework.

## Overview

tracking.py contains the implementation for the tracking task, which:
1. Identifies spots (single molecules) within image frames
2. Builds trajectories by linking these spots across multiple frames

## Main Function: `track(params)`

The primary function in this module is track(params), which deals with spot detection and trajectory construction process:

```python
def track(params):
    # Read in the image data
    image_data = images.ImageData()
    image_data.read(params.name + ".tif", params)
    
    # Process differently if ALEX (Alternating Laser Excitation) is used
    if params.ALEX==True:
        # ALEX processing code
        # ...
    else:
        # Standard processing
        all_spots = []
        if params.num_procs == 0:
            # Serial processing
            for frame in range(image_data.num_frames):
                all_spots.append(track_frame(image_data[frame], frame, params))
        else:
            # Parallel processing with multiprocessing
            res = [None] * image_data.num_frames
            with mp.Pool(params.num_procs) as pool:
                for frame in range(image_data.num_frames):
                    res[frame] = pool.apply_async(track_frame, (image_data[frame], frame, params))
                for frame in range(image_data.num_frames):
                    all_spots.append(res[frame].get())

        # Link the spot trajectories across frames
        trajs = trajectories.build_trajectories(all_spots, params)
        trajectories.write_trajectories(trajs, params.name + "_trajectoriesPY.csv")
```

### Key Sections:

1. **Image Loading**: 
   - Reads TIF image data using the `images.ImageData()` class

2. **Processing Path Selection**:
   - Handles differently based on whether ALEX mode is enabled
   - ALEX (Alternating Laser Excitation) is a technique that alternates excitation between two channels

3. **Spot Detection**:
   - Either processes frames serially or in parallel depending on `params.num_procs`
   - Each frame is processed by the `track_frame` function

4. **Trajectory Building**:
   - Links spots across frames to build trajectories using `trajectories.build_trajectories`
   - Saves the resulting trajectories to a CSV file

## Frame Processing: `track_frame(frame_data, frame, params)`

This function processes individual frames to detect and characterise spots:

```python
def track_frame(frame_data, frame, params):
    # Find the spots in this frame
    frame_spots = spots.Spots(frame=frame)
    frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
    found_spots = frame_spots.num_spots
    frame_spots.merge_coincident_candidates()

    merged_spots = frame_spots.num_spots
    # Iteratively refine the spot centres
    frame_spots.refine_centres(frame_data, params)

    frame_spots.filter_candidates(frame_data, params)

    frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
    frame_spots.get_spot_widths(frame_data.as_image()[:,:], params)
    if params.verbose:
        print(
            f"Frame {frame:4d}: found {frame_spots.num_spots:3d} spots "
            f"({found_spots:3d} identified, "
            f"{found_spots-merged_spots:3d} merged, "
            f"{merged_spots-frame_spots.num_spots:3d} filtered)"
        )
    return frame_spots
```

### Key Steps in Frame Processing:

1. **Spot Detection**: 
   - Creates a `Spots` object for the current frame
   - Uses `find_in_frame` to identify spots in the image

2. **Spot Refinement**:
   - `merge_coincident_candidates`: Merges spots that are too close together
   - `refine_centres`: Iteratively refines spot centers for improved precision
   - `filter_candidates`: Removes spots that don't meet quality criteria

3. **Spot Characterization**:
   - `get_spot_intensities`: Calculates intensity values for each spot
   - `get_spot_widths`: Determines the spatial extent of each spot

4. **Reporting**:
   - Logs progress information if verbose mode is enabled

## ALEX Mode Processing

When ALEX (Alternating Laser Excitation) is enabled, the code:

1. Separates the image data into left and right channels
2. Processes each channel independently
3. Creates separate trajectory files for each channel

## Integration with Other Modules

`tracking.py` interfaces with several other modules:

1. **images.py**: 
   - Provides the `ImageData` class for reading and handling image data

2. **spots.py**: 
   - Provides the `Spots` class for spot detection and characterization
   - Implements algorithms for finding, filtering, and measuring spots

3. **trajectories.py**: 
   - Provides functions for building trajectories from spots across frames
   - Handles writing trajectory data to files

## Workflow Summary

The overall flow can be summarised as:

1. Load image data from a TIF file
2. For each frame:
   - Detect spots using image processing techniques
   - Refine spot positions for higher precision
   - Filter out low-quality spots
   - Calculate spot properties (intensity, width)
3. Link spots across frames to build trajectories
4. Save trajectory data to a CSV file

## Parameters

Key parameters that influence the tracking process:

- **num_procs**: Number of parallel processes to use (0 for serial processing)
- **ALEX**: Whether to use Alternating Laser Excitation mode
- **struct_disk_radius**: Radius used for structural elements in image processing
- **filter_image**: Type of filtering to apply to images ("Gaussian" or "None")
- **bw_threshold_tolerance**: Threshold tolerance for binary image conversion
- **max_displacement**: Maximum allowed movement of a spot between frames
- **snr_filter_cutoff**: Signal-to-noise ratio cutoff for filtering spots
- **min_traj_len**: Minimum trajectory length to consider valid

These parameters are passed via the `params` object, which is an instance of the `Parameters` class defined in `parameters.py`.
