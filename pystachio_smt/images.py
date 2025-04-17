#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" IMAGES - Image access and manipulation module

Description:
    images.py contains the ImageData class for storing datasets of multiple
    frames, along with routines for manipulating the images.

Contains:
    class    ImageData
    function display_image

Author:
    Edward Higgins

Version: 0.2.0
"""

# --- Core library imports ---
import sys
import os
import numpy as np
import tifffile

# --- Class definition for managing image sequence data ---
class ImageData:
    # Initialise empty ImageData object
    def __init__(self):
        self.num_frames = -1
        self.has_mask = False
        self.pixel_data = None
        self.mask_data = None
        self.exists = False # Flag indicating if data is loaded

    # Allow accessing individual frames using slicing (e.g., image_data[0])
    def __getitem__(self, index):
        # Create a new ImageData object for the single frame
        frame = ImageData()
        # Initialise with 1 frame and same size
        frame.initialise(1, self.frame_size)
        # Copy pixel data for the requested frame
        frame.pixel_data[0, :, :] = self.pixel_data[index, :, :]
        # Copy mask information
        frame.mask_data = self.mask_data
        frame.has_mask = self.has_mask
        # Return the single-frame ImageData object
        return frame

    # Allow setting individual frames using slicing (e.g., image_data[0] = new_frame_data)
    def __setitem__(self, index, value):
        # If value is another ImageData object, copy its pixel data
        if isinstance(value, ImageData): # Check type properly
            self.pixel_data[index, :, :] = value.pixel_data[0, :, :]
        # Otherwise, assume value is a NumPy array and assign directly
        else:
            self.pixel_data[index, :, :] = value

    # Initialise object attributes for a given size
    def initialise(self, num_frames, frame_size):
        self.num_frames = num_frames
        self.frame_size = frame_size # (width, height)
        self.num_pixels = frame_size[0] * frame_size[1]

        # Create empty NumPy arrays for pixel and mask data
        self.pixel_data = np.zeros([num_frames, frame_size[1], frame_size[0]]) # Note: H, W order
        self.mask_data = np.zeros([frame_size[1], frame_size[0]]) # Mask assumed 2D for now
        self.has_mask = False

        # Set exists flag to True
        self.exists = True

    # Return pixel data as a NumPy array
    def as_image(self, frame=0, drop_dim=True):
        # Return single frame as 2D array if drop_dim is True
        if drop_dim:
            img = self.pixel_data.astype(np.uint16)[frame, :, :]
        # Return entire stack (or single frame as 3D) if drop_dim is False
        else:
            img = self.pixel_data.astype(np.uint16)
        return img

    # Read image data (TIFF) and mask (optional) from files
    def read(self, filename, params):
        # Check if image file exists
        if not os.path.isfile(filename):
            sys.exit(f"Unable to find file matching '{filename}'")

        # Read TIFF file using tifffile library
        pixel_data = tifffile.imread(filename)

        # Read and process optional cell mask
        if params.cell_mask:
            if os.path.isfile(params.cell_mask):
                pixel_mask = tifffile.imread(params.cell_mask)
                 # Ensure mask is binary (0 or 1) and matches frame size if 3D
                if pixel_mask.ndim == 3:
                    # Use first frame of mask if it's a stack, check size
                    if pixel_mask.shape[1:] == (pixel_data.shape[1], pixel_data.shape[2]):
                        pixel_mask = np.where(pixel_mask[0,:,:] > 0, 1, 0)
                    else:
                        sys.exit(f"ERROR: Mask dimensions {pixel_mask.shape[1:]} don't match image dimensions {(pixel_data.shape[1], pixel_data.shape[2])}")
                elif pixel_mask.ndim == 2:
                     # Use 2D mask directly if dimensions match
                     if pixel_mask.shape == (pixel_data.shape[1], pixel_data.shape[2]):
                         pixel_mask = np.where(pixel_mask > 0, 1, 0)
                     else:
                         sys.exit(f"ERROR: Mask dimensions {pixel_mask.shape} don't match image dimensions {(pixel_data.shape[1], pixel_data.shape[2])}")
                else:
                     sys.exit(f"ERROR: Unexpected mask dimension: {pixel_mask.ndim}")

                self.has_mask = True
                self.mask_data = pixel_mask # Store processed 2D mask
            else:
                print(f"Warning: Mask file not found: {params.cell_mask}")
                self.has_mask = False # Ensure flag is False if file not found
        else: # No mask provided
            self.has_mask = False
            # Create a default mask (all ones) matching the frame size later

        # Determine number of frames to use
        if params.num_frames:
            self.num_frames = min(params.num_frames, pixel_data.shape[0])
        else:
            self.num_frames = pixel_data.shape[0]

        # Determine frame size, handling channel splitting
        image_height = pixel_data.shape[1]
        image_width = pixel_data.shape[2]
        if params.channel_split == "Vertical":
            self.frame_size = (image_width // 2, image_height) # W, H
        elif params.channel_split == "Horizontal":
            self.frame_size = (image_width, image_height // 2) # W, H
        else: # No split
            self.frame_size = (image_width, image_height) # W, H

        # Calculate number of pixels per frame
        self.num_pixels = self.frame_size[0] * self.frame_size[1]

        # Store the selected frames and region
        # Ensure slicing uses correct height/width from self.frame_size
        self.pixel_data = pixel_data[:self.num_frames, :self.frame_size[1], :self.frame_size[0]]

        # Create default mask if none was loaded
        if not self.has_mask:
             self.mask_data = np.ones((self.frame_size[1], self.frame_size[0]), dtype=int) # H, W

        # Determine the first frame to process (placeholder method)
        self.determine_first_frame()

        # Mark object as containing data
        self.exists = True

    # Write image data to a TIFF file
    def write(self, filename):
        # Get image data as NumPy array (uint16)
        img_data = self.as_image(drop_dim=False)
        # Write using tifffile library
        tifffile.imwrite(filename, img_data)

    # Rotate image data (currently only handles 90-degree increments)
    def rotate(self, angle):
        # Check for valid rotation angle
        if angle % 90 == 0:
            # Rotate each frame individually
            for frame_idx in range(self.num_frames): # Corrected loop
                self.pixel_data[frame_idx, :, :] = np.rot90(self.pixel_data[frame_idx, :, :], angle // 90)
            # Update frame size if needed (e.g., 90/270 degree rotation swaps W/H)
            if (angle // 90) % 2 != 0:
                self.frame_size = (self.frame_size[1], self.frame_size[0]) # Swap W and H
        else:
            sys.exit("ERROR: Images can only be rotated by multiples of 90°")

    # Placeholder method to determine the first relevant frame
    def determine_first_frame(self):
        # Currently sets first frame to 0 unconditionally
        self.first_frame = 0 # Consider adding logic based on laser or intensity

    # Calculate the maximum intensity value across all frames
    def max_intensity(self):
        # Return max pixel value from the data
        max_intensity = np.max(self.pixel_data)
        return max_intensity

# """ IMAGES - Image access and manipulation module

# Description:
#     images.py contains the ImageData class for storing datasets of multiple
#     frames, along with routines for manipulating the images.

# Contains:
#     class    ImageData
#     function display_image

# Author:
#     Edward Higgins

# Version: 0.2.0
# """

# import sys
# import os

# import cv2 as cv
# import matplotlib.animation as animation
# #import matplotlib.pyplot as plt
# import numpy as np
# import tifffile


# class ImageData:
#     def __init__(self):
#         self.num_frames = -1
#         self.has_mask = False
#         self.pixel_data = None
#         self.mask_data = None
#         self.exists = False

#     def __getitem__(self, index):
#         frame = ImageData()
#         frame.initialise(1, self.frame_size)
#         frame.pixel_data[0, :, :] = self.pixel_data[index, :, :]
#         frame.mask_data = self.mask_data
#         frame.has_mask = self.has_mask
#         return frame

#     def __setitem__(self, index, value):
#         if value.__class__ == "ImageData":
#             self.pixel_data[index, :, :] = value.pixel_data[0, :, :]
#         else:
#             self.pixel_data[index, :, :] = value

#     def initialise(self, num_frames, frame_size):
#         self.num_frames = num_frames
#         self.frame_size = frame_size
#         self.num_pixels = frame_size[0] * frame_size[1]

#         self.pixel_data = np.zeros([num_frames, frame_size[1], frame_size[0]])
#         self.mask_data = np.zeros([num_frames, frame_size[1], frame_size[0]])
#         self.has_mask = False

#         self.exists = True

#     def as_image(self, frame=0, drop_dim=True):

#         if drop_dim:
#             img = self.pixel_data.astype(np.uint16)[frame, :, :]
#         else:
#             img = self.pixel_data.astype(np.uint16)
#         return img

#     def read(self, filename, params):
#         # Determine the filename from the seedname
#         if not os.path.isfile(filename):
#             sys.exit(f"Unable to find file matching '{filename}'")

#         # Read in the file and get the data size
#         pixel_data = tifffile.imread(filename)

#         if params.cell_mask:
#             if os.path.isfile(params.cell_mask):
#                 pixel_mask = tifffile.imread(params.cell_mask)
#                 pixel_mask = np.where(pixel_mask > 0, 1, 0)
#                 self.has_mask = True
#                 self.mask_data = pixel_mask
#         else:
#             self.use_mask = False

#         if params.num_frames:
#             self.num_frames = min(params.num_frames, pixel_data.shape[0])
#         else:
#             self.num_frames = pixel_data.shape[0]

#         if params.channel_split == "Vertical":
#             self.frame_size = (pixel_data.shape[2]//2, pixel_data.shape[1])
#         if params.channel_split == "Horizontal":
#             self.frame_size = (pixel_data.shape[2], pixel_data.shape[1]//2)
#         else:
#             self.frame_size = (pixel_data.shape[2], pixel_data.shape[1])

#         self.num_pixels = self.frame_size[0] * self.frame_size[1]

#         # Store the frames in a list
#         self.pixel_data = pixel_data[:self.num_frames, :self.frame_size[1], :self.frame_size[0]]

#         self.determine_first_frame()

#         self.exists = True

#     def write(self, filename):
#         # Create the data array
#         img_data = self.as_image(drop_dim=False)

#         tifffile.imwrite(filename, img_data)

#     def rotate(self, angle):
#         if angle % 90 == 0:
#             for frame in self.num_frames:
#                 np.rot90(self.pixel_data[frame, :, :], angle // 90)

#         else:
#             sys.exit("ERROR: Images can only be rotated by multiples of 90°")

#     def determine_first_frame(self):
#         self.first_frame = 0

#     def max_intensity(self):
#         max_intensity = np.max(self.pixel_data)

#         return max_intensity
