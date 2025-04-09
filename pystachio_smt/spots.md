# Understanding `spots.py`: Single Molecule Fluorescence Spot Detection and Analysis

The spots.py module is responsible for the detection, characterisation, and management of spots in images.

## Overview

The `spots.py` module contains the `Spots` class, which handles all aspects of spot detection and analysis. It employs sophisticated image processing techniques to precisely locate single molecule spots and characterise their properties.

## The `Spots` Class

The `Spots` class represents a collection of fluorescent spots within a single frame of microscopy data. It provides methods for:

1. Finding spots in an image
2. Refining spot positions to sub-pixel accuracy
3. Measuring spot intensities and sizes
4. Filtering out low-quality spot detections
5. Supporting trajectory linking between frames

### Key Attributes

* `num_spots`: Number of spots being tracked
* `positions`: Array of [x,y] coordinates for each spot
* `spot_intensity`: Array of intensity values for each spot
* `bg_intensity`: Array of background intensity values
* `snr`: Array of signal-to-noise ratios
* `width`: Array of spot width measurements
* `traj_num`: Array assigning each spot to a trajectory
* `converged`: Flags indicating successful position refinement

## Key Methods

### Initialisation

```python
def __init__(self, num_spots=0, frame=0):
```

The constructor creates a new `Spots` object with:
* Optional pre-allocation for a specified number of spots
* Frame number assignment
* Initialisation of data arrays if spots are being tracked

### Position Management

```python
def set_positions(self, positions):
```

This method:
* Updates the object with new spot positions
* Resets all associated data arrays
* Creates consistent data structures for spot analysis

### Spot Detection

```python
def find_in_frame(self, frame, params):
```

This is the primary spot detection method that implements a multi-step image processing workflow:

1. **Image Preparation**:
   * Converts the greyscale frame to BGR format
   * Creates a disk-shaped structuring element for morphological operations
   * Optionally applies Gaussian filtering to reduce noise

2. **Spot Enhancement**:
   * Applies top-hat morphological filtering to highlight small bright features
   * Calculates histogram to determine threshold values
   * Applies additional Gaussian filtering
   * Creates binary image via thresholding

3. **Spot Refinement**:
   * Performs morphological opening to remove noise
   * Fills small holes in detected regions
   * Uses ultimate erosion algorithm to find spot centres
   * Sets detected positions in the object

### Spot Post-Processing

```python
def merge_coincident_candidates(self):
```

This method identifies and merges spots that are too close together (less than 2 pixels apart) by averaging their positions.

```python
def filter_candidates(self, frame, params):
```

Removes low-quality spot detections based on:
* Signal-to-noise ratio below threshold
* Spots outside any mask region
* Spots too close to image edges

### Intensity and Size Measurement

```python
def get_spot_intensities(self, frame, params):
```

For each spot:
* Extracts a small sub-image around the spot
* Creates a circular mask
* Calculates local background intensity
* Measures background-corrected spot intensity

```python
def get_spot_widths(self, frame, params):
```

Measures spot dimensions by:
* Extracting a region around each spot
* Subtracting background
* Fitting a 2D Gaussian function to determine width

### Position Refinement

```python
def refine_centres(self, frame, params):
```

Implements an iterative algorithm to achieve sub-pixel localisation accuracy:

1. For each spot, extract a sub-region around the initial position
2. Create both circular and Gaussian weighting masks
3. Calculate and subtract local background
4. Compute intensity-weighted centroid for improved position estimate
5. Repeat until position converges or maximum iterations reached

This process achieves nanometre-scale localisation precision from diffraction-limited images.

### Trajectory Support

```python
def distance_from(self, other):
```

Calculates pairwise distances between spots in different frames to support trajectory linking.

```python
def link(self, prev_spots, params):
```

Links spots with those in the previous frame, assigning trajectory numbers based on proximity.

## Image Processing Techniques Used

The module employs several advanced image processing techniques:

1. **Morphological Operations**:
   * Top-hat filtering to enhance small bright features
   * Opening and closing for noise removal and feature preservation

2. **Gaussian Fitting**:
   * 2D Gaussian model fitting for accurate spot characterisation
   * Gaussian weighting for centroid determination

3. **Background Estimation**:
   * Local background calculation for each spot
   * Background subtraction for accurate intensity measurement

4. **Thresholding**:
   * Adaptive thresholding based on image histograms
   * Binary image creation for feature isolation

5. **Centroid Analysis**:
   * Intensity-weighted centroid calculation for sub-pixel localisation
   * Iterative refinement for improved accuracy

## Integration with the Tracking System

The `Spots` class serves as the foundation for trajectory building in the SMT system:

1. For each frame, spots are detected and characterised
2. Spot positions are refined to sub-pixel accuracy
3. Spots are linked across frames based on proximity
4. Trajectories are constructed from these linked spots

The quality of trajectory analysis depends critically on the accuracy of the initial spot detection and characterisation, making this module essential to the overall system performance.

## Performance Considerations

Several parameters influence the spot detection performance:

* `struct_disk_radius`: Size of the structuring element for morphological operations
* `filter_image`: Type of pre-filtering applied
* `bw_threshold_tolerance`: Threshold sensitivity for binary image creation
* `snr_filter_cutoff`: Minimum signal-to-noise ratio for valid spots
* `inner_mask_radius`: Size of spot integration region
* `gauss_mask_sigma`: Width of Gaussian for position refinement
* `subarray_halfwidth`: Size of region extracted around each spot

These parameters can be tuned to optimise detection for different experimental conditions.

## Typical Workflow

A typical processing sequence for a single frame:

1. **Spot Detection**: `find_in_frame()` locates potential spots
2. **Candidate Merging**: `merge_coincident_candidates()` combines close spots
3. **Position Refinement**: `refine_centres()` improves localisation precision
4. **Candidate Filtering**: `filter_candidates()` removes low-quality detections
5. **Intensity Measurement**: `get_spot_intensities()` calculates spot brightnesses
6. **Size Measurement**: `get_spot_widths()` determines spot dimensions

This comprehensive approach ensures accurate detection and characterisation of single molecule fluorescent spots, providing the foundation for reliable single molecule tracking.
