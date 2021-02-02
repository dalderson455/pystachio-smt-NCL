#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

"""
Single Molecule Tools
"""
import sys
import numpy as np

import tracking
import simulation
import parameters
import images
import postprocessing

def main():
    params = parameters.Parameters()
    params.read(sys.argv)

    for task in params.task:
        if task == "track":
            image_data = images.ImageData()
            image_data.read(params.seed_name + ".tif")

            if params.verbose:
                print(f"Loaded {image_data.num_frames} frames from {params.seed_name}")
                print(f"Resolution: {image_data.resolution}")
                
            spots, trajs = tracking.track(image_data, params)
            intensities = np.array([])
            for i in range(len(spots)):
                tmp = spots[i].spot_intensity
                intensities = np.concatenate((intensities,tmp))
            postprocessing.get_isingle(intensities)
            diff_coef, diff_coef_loc_precision  = postprocessing.get_diffusion_coef(trajs, params)
            print(np.mean(diff_coef))
            postprocessing.plot_traj_intensities(spots)
            
        elif task == "simulate":
            image_data = simulation.simulate(params)
#EJH#             spot_data.write(params)
            image_data.write(params)

        elif task=="simulate_stepwise":
            image_data = simulation.simulate_stepwise_bleaching(params)
            image_data.write(params)
    
        elif task == "view":
            img = images.ImageData()
            img.read(params.seed_name + ".tif")
            img.render(params)

        else:
            sys.exit(f"ERROR: Task {task} is not yet implemented. Aborting...")


if __name__ == "__main__":
    main()
