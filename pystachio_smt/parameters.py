#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" PARAMETERS - Program parameters module

Description:
    parameters.py contains the Parameters class that holds all the program
    parameters, along with the default value for each parameter and routines
    for setting those parameters.

Contains:
    class Parameters

Author:
    Edward Higgins

Version: 0.2.0
"""

import sys
from difflib import SequenceMatcher

default_parameters = {
    # Runtime parameters
    'display_figures':
        { 'description': 'Figures (Y/n)',
          'level': 'basic',
          'class': 'general',
          'default': True },    
    'num_procs':
        { 'description': 'The number of CPU processes to run with (0 is serial)',
          'level': 'basic',
          'class': 'general',
          'default': 0 },
    'tasks':
        { 'description': 'Which task(s) to perform in the run',
          'level': 'basic',
          'class': 'general',
          'default': [],
          'options': ['simulate', 'track', 'postprocess', 'view'] },
    'name':
        { 'description': 'Name prefixing all files associated with this run',
          'level': 'basic',
          'class': 'general',
          'default': '' },
    'mask_file':
        { 'description': 'Filename of an image mask for filtering spots',
          'level': 'basic',
          'class': 'general',
          'default':  ''},
    'verbose':
        { 'description': 'Whether or not to display progress messages',
          'level': 'basic',
          'class': 'general',
          'default':  True},

    # Image parameters
    'frame_time':
        { 'description': 'Time per frame in seconds',
          'level': 'advanced',
          'class': 'image',
          'default': 0.005 },
    'pixel_size':
        { 'description': 'Length of a single pixel in μm',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'psf_width':
        { 'description': '?',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'start_frame':
        { 'description': 'The first frame of the image stack to analyse',
          'level': 'basic',
          'class': 'image',
          'default':  0},
    'num_frames':
        { 'description': 'Number of frames to simulate/analyse',
          'level': 'basic',
          'class': 'image',
          'default': 0 },
    'channel_split':
        { 'description': 'If/how the frames are split spatially',
          'level': 'basic',
          'class': 'image',
          'default': 'None',
          'options': ['None', 'Vertical', 'Horizontal'] },
    'cell_mask':
        { 'description': 'Name of a black/white TIF file containing a cell mask',
          'level': 'advanced',
          'class': 'image',
          'default': ''},
    'ALEX':
        { 'description': 'Perform Alternating-Laser experiment analysis',
          'level': 'basic',
          'class': 'image',
          'default': False},
    'start_channel':
        { 'description': '?',
          'level': 'basic',
          'class': 'image',
          'default': 'L'},

    # Simulation parameters
    'num_spots':
        { 'description': 'Number of spots to simulate',
          'level': 'basic',
          'class': 'simulation',
          'default': 50 },
    'frame_size':
        { 'description': 'Size of frame to simulate ([x,y])',
          'level': 'basic',
          'class': 'simulation',
          'default': [250,250] },
    'I_single':
        { 'description': 'I_single value for simulated spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 10000.0 },
    'bg_mean':
        { 'description': 'Mean of the background pixel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 500.0 },
    'bg_std':
        { 'description': 'Standard deviation of the background pixel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 120.0 },
    'diffusion_coeff':
        { 'description': 'Diffusion coefficient of the diffusing spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 2.0 },
    'spot_width':
        { 'description': 'Width of the simulated Gaussian spot',
          'level': 'advanced',
          'class': 'simulation',
          'default':  1.33},
    'max_spot_molecules':
        { 'description': 'Maximum number of dye molecules per spot',
          'level': 'advanced',
          'class': 'simulation',
          'default': 1 },
    'p_bleach_per_frame':
        { 'description': 'Probability of a spot bleaching in a given frame',
          'level': 'advanced',
          'class': 'simulation',
          'default': 0.0 },
    'photobleach':
        { 'description': 'Perform photobleaching (alias for max_spot_molecules=10, p_bleach_per_frame=0.05)',
          'level': 'basic',
          'class': 'simulation',
          'default': False },

    # Tracking parameters
    'bw_threshold_tolerance':
        { 'description': 'Threshold for generating the b/w image relative to the peak intensity',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1.0 },
    'snr_filter_cutoff':
        { 'description': 'Cutoff value when filtering spots by signal/noise ratio',
          'level': 'basic',
          'class': 'tracking',
          'default': 0.4 },
    'filter_image':
        { 'description': 'Method for filtering the input image pre-analysis',
          'level': 'advanced',
          'class': 'tracking',
          'default': 'Gaussian',
          'options': ['Gaussian', 'None']},
    'max_displacement':
        { 'description': 'Maximum displacement allowed for spots between frames',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5.0 },
    'struct_disk_radius':
        { 'description': 'Radius of the Disk structural element',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },
    'min_traj_len':
        { 'description': 'Minimum number of frames needed to define a trajectory',
          'level': 'advanced',
          'class': 'tracking',
          'default': 3 },
    'subarray_halfwidth':
        { 'description': 'Halfwidth of the sub-image for analysing individual spots',
          'level': 'advanced',
          'class': 'tracking',
          'default': 8 },
    'gauss_mask_sigma':
        { 'description': 'Width of the Gaussian used for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 2.0 },
    'gauss_mask_max_iter':
        { 'description': 'Max number of iterations for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1000 },
    'inner_mask_radius':
        { 'description': 'Radius of the mask used for calculating spot intensities',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },

    # Postprocessing parameters
    'display_figures':
        { 'description': 'Whether or not to display the figures live in MatPlotLib',
          'level': 'basic',
          'class': 'postprocessing',
          'default': True},
    'chung_kennedy_window':
        { 'description': 'Window width for Chung-Kennedy filtering',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 3},
    'chung_kennedy':
        { 'description': 'Flag to specify whether or not to Chung-Kennedy filter intensity tracks',
          'level': 'basic',
          'class': 'postprocessing',
          'default': True},
    'msd_num_points':
        { 'description': 'Number of points used to calculate the mean-squared displacement',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 4 },
    'stoic_method':
        { 'description': 'Method used for determining the stoichiometry of each trajectory',
          'level': 'advanced',
          'class': 'postprocessing',
          'default': 'Linear',
          'options': ['Linear', 'Mean', 'Initial'] },
    'num_stoic_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Number of frames used to determine the stoichiometry',
          'default': 3 },
    'calculate_isingle': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Whether or not to calculate the ISingle',
          'default': True },
    'colocalize': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
    'colocalize_distance': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'colocalize_n_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'copy_number': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
}


class Parameters:
    def __init__(self, initial=default_parameters):
        self._params = initial

        for param in self._params.keys():
            # Set all the values to be the default values
            self._params[param]['value'] = self._params[param]['default']

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        else:
            try:
                return object.__getattribute__(self, "_params")[name]['value']
            except KeyError as exc:
                max_param = ''
                max_val  = 0
                for key in self._params:
                    if SequenceMatcher(None, name, key).ratio() > max_val:
                        max_param = key
                        max_val = SequenceMatcher(None, name, key).ratio()
                print(f"\nNo such key {name}. Did you mean {max_param}?\n")
                raise  exc

    def __setattribute__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._params[name]['value'] = value

    def help(self, name=None, param_class=None, level='basic'):
        names = []
        if name:
            names.append(name)
        elif param_class:
            for key in self._params:
                if level != 'advanced':
                    if (self._params[key]['class'] == param_class
                      and self._params[key]['level'] == 'basic'):
                        names.append(key)
                else:
                    names.append(key)
        elif level != 'basic':
            for key in self._params:
                names.append(key)
        else:
            for key in self._params:
                if self._params[key]['level'] == 'basic':
                    names.append(key)


        for name in names:
            print()
            print(f"{name.upper()}")
            print(f"  Description: {self._params[name]['description']}")
            print(f"      Default: {self._params[name]['default']}")
            print(f"        Class: {self._params[name]['class']}")
            print(f"        Level: {self._params[name]['level']}")



    def read(self, args):
        self.task = args.pop(0)
        self.task = self.task.split(",")
        if self.task == ['help']:
            return
        elif self.task != ['app']:
             self.name = args.pop(0)

        for arg in args:
            key, value = arg.split("=", 2)
            try:
                # use isinstance
                if type(getattr(self, key)) is type(0):
                    setattr(self, key, int(value))

                elif type(getattr(self, key)) is type(0.0):
                    setattr(self, key, float(value))

                elif type(getattr(self, key)) is type(True):
                    setattr(self, key, value == "True")

                elif type(getattr(self, key)) is type([]):
                    setattr(self, key, list(map(lambda x: int(x), value.split(","))))

                else:
                    setattr(self, key, value)

            except NameError:
                sys.exit(f"ERROR: No such parameter '{key}'")

            if key == "pixel_size":
                self.psf_width = 0.160 / self.pixel_size

    def param_dict(self, param_class=''):
        param_dict = {}

        
        if param_class:
            for k,v in self._params.items():
                if v["class"] == param_class:
                    param_dict[k] = v
        else:
            param_dict = self._params

        return param_dict


