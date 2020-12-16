#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

"""
IMAGES
"""

import tifffile
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 as cv

class ImageData():
    def __init__(self):
        self.defined = False

    def __getitem__(self, index):
        frame = ImageData()
        frame.initialise(1, self.resolution)
        frame.pixel_data[0,:,:] = self.pixel_data[index,:,:]
        return frame

    def initialise(self, num_frames, resolution):
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_pixels = resolution[0] * resolution[1]

        self.pixel_data = np.zeros([num_frames, resolution[0], resolution[1]])

        self.defined = True

    def as_image(self):
        max_val = np.max(self.pixel_data)
        min_val = np.min(self.pixel_data)

        img = (256 * (self.pixel_data[0,:,:]+min_val) / (max_val+min_val)).astype(np.uint8)
        return img

    def read(self, filename):
        # Read in the file and get the data size
        pixel_data = tifffile.imread(filename)
        self.num_frames = pixel_data.shape[0]
        self.resolution = (pixel_data.shape[1], pixel_data.shape[2])
        self.num_pixels = pixel_data.shape[1] * pixel_data.shape[2]

        # Store the frames in a list
        self.pixel_data = pixel_data

        self.determine_first_frame()

        self.defined = True

    def write(self, params):
        # Create the data array

        tifffile.imsave(params.filename, self.pixel_data)

    def rotate(self, angle):
        if angle % 90 == 0:
            for frame in self.num_frames:
                np.rot90(self.pixel_data[frame,:,:], angle//90)

        else:
            sys.exit("ERROR: Images can only be rotated by multiples of 90°")

    def determine_first_frame(self):
        self.first_frame = 0

    def max_intensity(self):
        max_intensity = np.max(self.pixel_data)

        return max_intensity

    def render(self):
        print("Rendering image")
        maxval = self.max_intensity()
        figure = plt.figure()
        plt_frames = []

        for frame in range(self.num_frames):
            plt_frame = plt.imshow(self.pixel_data[frame,:,:], animated=True, vmin=0, vmax=maxval)
            plt_frames.append([plt_frame])

        video = animation.ArtistAnimation(figure, plt_frames, interval=50)
        plt.show()

        print("Done!")

def display_image(img):
    cv.imshow('image', img)
    cv.watKEy(0)
    cv.destroyAllWindows()
