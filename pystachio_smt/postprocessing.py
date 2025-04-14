"""
Post processing trajectories and intensities

Routines for getting isingle and diffusion coefficient from lists of intensities

v0.1 Jack W Shepherd, University of York
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.spatial import distance_matrix
import os
# Import local modules
from . import trajectories, images

# Global flag for displaying figures (currently off by default)
display_figures = False

def postprocess(params, simulated=False):
    # Set display flag based on params
    display_figures = params.display_figures

    # Check if it's NOT an ALEX experiment
    if not params.ALEX:
        # Initialise trajectory list
        trajs = []
        # Decide whether to read simulated or tracked trajectories
        # Note: 'if False:' condition currently prevents reading simulated trajectories
        if False: #simulated:
            trajs = trajectories.read_trajectories(params.name + "_simulated_trajectories.csv")
        else:
            trajs = trajectories.read_trajectories(params.name + "_trajectories.csv")

        # Convert trajectories list to spots list
        spots = trajectories.to_spots(trajs)
        # Print info if verbose mode is on
        if params.verbose:
            print(f"Looking at {len(trajs)} trajectories across {len(spots)} frames")

        # Initialise arrays for intensity and SNR
        intensities = np.array([])
        snrs = np.array([])
        # Loop through spots in each frame to collect intensities and SNRs
        for i in range(len(spots)):
            tmp = spots[i].spot_intensity
            tmp_snr = spots[i].snr
            intensities = np.concatenate((intensities,tmp))
            snrs = np.concatenate((snrs,tmp_snr))

        # Calculate or use predefined single molecule intensity (isingle)
        if params.calculate_isingle:
            calculated_isingle = get_isingle(params,intensities)
        else:
            calculated_isingle = params.I_single
        # Plot SNR distribution and get peak SNR
        calculated_snr = plot_snr(params,snrs)
        # Get diffusion coefficient (dc) and localisation precision (lp)
        dc, lp = get_diffusion_coef(trajs, params)

        # Print results if verbose mode is on
        if simulated:
            if params.verbose:
                print(f"Simulated diffusion coefficient: {np.mean(dc)}")
                print(f"Simulated Isingle:               {calculated_isingle[0]}")
        else:
            if params.verbose:
                print(f"Tracked diffusion coefficient: {np.mean(dc)}")
            #print(f"Tracked Isingle:               {calculated_isingle[0]}") # Isingle print commented out

        # Plot trajectory intensities (optionally applying Chung-Kennedy filter)
        plot_traj_intensities(params, trajs, params.chung_kennedy)
        # Calculate stoichiometries
        get_stoichiometries(trajs, calculated_isingle, params)
        # Calculate copy number if requested
        if params.copy_number==True: get_copy_number(params, calculated_isingle)

    # Check IF it IS an ALEX experiment
    elif params.ALEX:
        # Read trajectories and convert to spots for R and L channels if colocalisation needed
        if params.colocalize==True:
            Rtrajs = trajectories.read_trajectories(params.name + "_Rchannel_trajectories.csv")
            Ltrajs = trajectories.read_trajectories(params.name + "_Lchannel_trajectories.csv")
            Rspots = trajectories.to_spots(Rtrajs)
            Lspots = trajectories.to_spots(Ltrajs)

        # Initialise arrays for R and L channel intensities and SNRs
        Rintensities= np.array([])
        Rsnrs = np.array([])
        Lintensities= np.array([])
        Lsnrs = np.array([])
        # Collect R channel intensities and SNRs
        for i in range(len(Rspots)):
            Rintensities = np.concatenate((Rintensities,Rspots[i].spot_intensity))
            Rsnrs = np.concatenate((Rsnrs,Rspots[i].snr))
        # Collect L channel intensities and SNRs
        for i in range(len(Lspots)):
            Lintensities = np.concatenate((Lintensities,Lspots[i].spot_intensity))
            Lsnrs = np.concatenate((Lsnrs,Lspots[i].snr))
        # Calculate or use predefined isingle for R and L channels
        if params.calculate_isingle:
            R_isingle = get_isingle(params,Rintensities, channel="R")
            L_isingle = get_isingle(params,Lintensities, channel="L")
        else:
            R_isingle = params.R_isingle
            L_isingle = params.L_isingle

        # Plot SNR, get diffusion coeffs, plot intensities, get stoichiometries for L channel
        L_calculated_snr = plot_snr(params,Lsnrs, channel="L")
        Ldc, Llp = get_diffusion_coef(Ltrajs, params, channel="L")
        plot_traj_intensities(params, Ltrajs, channel="L")
        get_stoichiometries(Ltrajs, L_isingle, params, channel="L")

        # Plot SNR, get diffusion coeffs, plot intensities, get stoichiometries for R channel
        R_calculated_snr = plot_snr(params,Rsnrs, channel="R")
        Rdc, Rlp = get_diffusion_coef(Rtrajs, params, channel="R")
        plot_traj_intensities(params, Rtrajs, channel="R")
        get_stoichiometries(Rtrajs, R_isingle, params, channel="R")

        # Calculate copy numbers for L and R channels if requested
        if params.copy_number==True:
            get_copy_number(params, L_isingle, channel="L") # Using isingle, was calculated_isingle
            get_copy_number(params, R_isingle, channel="R") # Using isingle, was calculated_isingle
        # Perform colocalisation analysis if requested
        if params.colocalize==True:
            colocalize(params, Ltrajs, Rtrajs)

    # Exit if ALEX parameter is not True or False
    else: sys.exit("ERROR: look do you want ALEX or not?\nSet params.ALEX=True or False")

def chung_kennedy_filter(data, window, R): # Function applies Chung-Kennedy filter
    # This implementation is based on existing Fortran code.
    # Get data length
    N = len(data)
    # Create extended array padded with reflected data at ends
    extended_data = np.zeros(N+2*window)
    extended_data[window:-window]=data
    for i in range(1,window+1):
        extended_data[window-i] = data[i-1] # Reflect start
        extended_data[N+window+i-1] = data[N-i] # Reflect end
    # Get extended length
    N_extended = extended_data.shape[0]
    # Initialise arrays
    wdiffx = np.zeros(N_extended)
    datamx = np.zeros((N+window,window))
    datx = np.zeros((N+window+1,window))
    # Populate datamx with sliding windows
    for i in range(window):
        stop = N+window+i
        datamx[:,i] = extended_data[i:stop]
    # Calculate mean (wx) and std dev (sx) across windows
    wx = np.mean(datamx,axis=1)
    sx = np.std(datamx,axis=1,ddof=1) # ddof=1 for sample std dev
    # Prepare forward (XP) and backward (XM) means
    XP = wx[:N_extended]
    XM = wx[window+1:N_extended+window+1]
    # Prepare forward (SDP) and backward (SDM) std devs
    SDP = sx[:N]
    SDM = sx[window:N_extended+window]
    # Difference in std devs
    DSD = SDP-SDM
    # Squared std devs
    SP = SDP**2
    SM = SDM**2

    # Calculate switching functions based on variance ratio (RSP, RSM)
    RSP = SP**R
    RSM = SM**R
    # Avoid division by zero if RSP+RSM is zero
    GM = np.divide(RSP, RSP + RSM, out=np.zeros_like(RSP), where=(RSP + RSM) != 0)
    GP = np.divide(RSM, RSP + RSM, out=np.zeros_like(RSM), where=(RSP + RSM) != 0)
    # GM = RSP/(RSP+RSM) # Original, risks division by zero
    # GP = RSM/(RSP+RSM) # Original, risks division by zero

    # Initialise output arrays
    S = np.zeros(GP.shape)
    XX = np.zeros(GP.shape)
    # Calculate filtered variance (S) and filtered data (XX) using switching funcs
    for i in range(len(GP)-1):
        # Check if switching functions are within valid range [0, 1]
        if GM[i]>=0 and GM[i]<=1 and GP[i]>=0 and GP[i]<=1:
            S[i] = GM[i]*SM[i] + GP[i]*SP[i] # Weighted variance
            XX[i] = GP[i]*XP[i] + GM[i]*XM[i] # Weighted mean
        else:
            # Fallback if switching functions are invalid
            S[i] =  SP[i]
            XX[i] = XP[i]

    # Return only the filtered data XX (original code returned more)
    return [XX]

def colocalize(params, Ltrajs, Rtrajs): # Function to find colocalised spots between L and R channels
    # Read image data again (needed for plotting)
    image_data = images.ImageData()
    image_data.read(params.name + '.tif', params)
    # Create arrays for L and R channel images
    imageL=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
    imageR=np.zeros((image_data.num_frames//2,image_data.frame_size[1],image_data.frame_size[0]//2))
    # Split frames into L and R based on starting channel
    if params.start_channel=='L':
        for i in range(0,image_data.num_frames-1,2):
            imageL[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
            imageR[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]
    else:
        for i in range(0,image_data.num_frames-1,2):
            imageR[i//2,:,:] = image_data.pixel_data[i,:,:image_data.frame_size[0]//2]
            imageL[i//2,:,:] = image_data.pixel_data[i+1,:,image_data.frame_size[0]//2:]

    # Initialise lists to store linked trajectory IDs and link counts
    Llinks = []
    Rlinks = []
    nlinks = []
    # Iterate through each frame
    for i in range(params.num_frames): # Potential issue: uses params.num_frames, should use imageL.shape[0]?
        # Collect spots present in the current frame for L and R channels
        s1 = [] # L channel spots [position, width, id]
        s2 = [] # R channel spots [position, width, id]
        for traj in Ltrajs:
            if traj.start_frame<=i and traj.end_frame>=i:
                s1.append([traj.path[i-traj.start_frame],traj.width,traj.id])
        for traj in Rtrajs:
            if traj.start_frame<=i and traj.end_frame>=i:
                s2.append([traj.path[i-traj.start_frame],traj.width,traj.id])

        # Link spots between channels for the current frame
        id1, id2 = linker(params,s1,s2) # Get linked L_id (id1) and R_id (id2)

        # Update the overall link lists (Llinks, Rlinks, nlinks)
        for j in range(len(id1)):
            found = False
            # Check if this L_id is already in Llinks
            if id1[j] in Llinks: # Potential error: id1[i] should be id1[j]
                 indices = np.where(np.array(Llinks) == id1[j])[0] # Find indices matching L_id
                 for index in indices:
                     # If the corresponding R_id also matches, increment count
                     if Rlinks[index]==id2[j]:
                         nlinks[index] += 1
                         found = True
                         break # Found the existing pair
            # If pair not found, add as a new link
            if found == False:
                Llinks.append(id1[j])
                Rlinks.append(id2[j])
                nlinks.append(1) # Start count at 1

    # Write colocalised trajectories (meeting min frame requirement) to file
    outfile = params.name + "_colocalized_trajectories.csv"
    f = open(outfile, 'w')
    f.write("Left_traj\tRight_traj\tN_frames\n") # Header, corrected typo
    for i in range(len(Llinks)):
        if nlinks[i]>=params.colocalize_n_frames:
            f.write(f"{Llinks[i]}\t{Rlinks[i]}\t{nlinks[i]}\n") # Write data
    f.close()

    # --- Plotting Section ---
    # Define colours for plotting
    colors = ['r','b','g','m','y','c']
    c = 0 # Colour index
    # Get image dimensions
    z, y, x = imageL.shape
    # Create combined L/R display frame
    disp_frame = np.zeros((y,2*x))
    disp_frame[:,:x] = imageL[1,:,:] # Show frame 1 (index 1)
    disp_frame[:,x:] = imageR[1,:,:] # Show frame 1 (index 1)

    # Plot combined image background
    plt.imshow(disp_frame, cmap="Greys_r")
    # Draw dividing line
    plt.plot([x,x],[0,x], 'w--')
    # Formatting
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0,x]) # Assuming square aspect ratio for ylim
    # Show plot if enabled
    if params.display_figures: # Corrected variable name
        plt.show()

    # Plot colocalised spots on top of the image
    plt.imshow(disp_frame, cmap="Greys_r") # Re-plot background
    # Get IDs of L trajectories that were successfully linked in the linker function call for the *last processed frame*
    linked_l_ids_last_frame = id1 # Note: This only uses links from the last frame processed in the loop above
    # Iterate through all L trajectories
    for traj in Ltrajs:
        # Check if this L trajectory ID was linked in the last frame
        if traj.id in linked_l_ids_last_frame:
            # Find the index of this L_id in the last frame's links
            link_index_last_frame = np.where(linked_l_ids_last_frame == traj.id)[0]
            if len(link_index_last_frame) > 0:
                link_index_last_frame = link_index_last_frame[0] # Take first match if multiple
                 # Get the corresponding R trajectory ID from the last frame's links
                linked_r_id_last_frame = id2[link_index_last_frame]

                # Find the actual R trajectory object with this ID
                found_r_traj = None
                for t2 in Rtrajs:
                    if t2.id == linked_r_id_last_frame:
                        found_r_traj = t2
                        break

                # If corresponding R trajectory found, plot L and R spots
                if found_r_traj is not None:
                     # Check if spots exist in frame 1 (index 1)
                     l_frame_index = 1 - traj.start_frame
                     r_frame_index = 1 - found_r_traj.start_frame
                     if 0 <= l_frame_index < traj.length and 0 <= r_frame_index < found_r_traj.length:
                         # Plot L spot (adjusting for 0-based index)
                        plt.scatter(traj.path[l_frame_index][0], traj.path[l_frame_index][1], 10, marker='x', color=colors[c % len(colors)])
                         # Plot R spot (adjusting for display offset and 0-based index)
                        plt.scatter(found_r_traj.path[r_frame_index][0]+x, found_r_traj.path[r_frame_index][1], 10, marker='x', color=colors[c % len(colors)])
                        c+=1 # Increment colour index only if plotted

    # Final plot formatting
    plt.yticks([])
    plt.xticks([])
    plt.plot([x,x],[0,x], 'w--') # Redraw dividing line
    plt.ylim([0,x]) # Assuming square aspect ratio
    plt.savefig("colocalized_spots.png", dpi=300)
    # Show plot if enabled
    if params.display_figures: # Corrected variable name
        plt.show()
    plt.close()

def linker(params, spots1, spots2): # Links spots between two lists based on distance and Gaussian overlap
    # Extract positions and widths from input spot lists
    s1_pos = []
    s2_pos = []
    s1_width = []
    s2_width = []
    # Populate lists for spots1 (L channel)
    for i in range(len(spots1)): # Iterate through spots in list 1
        s1_pos.append(spots1[i][0]) # Append position [x, y]
        s1_width.append(spots1[i][1]) # Append width [wx, wy]
    # Populate lists for spots2 (R channel)
    for i in range(len(spots2)): # Iterate through spots in list 2
        s2_pos.append(spots2[i][0]) # Append position [x, y]
        s2_width.append(spots2[i][1]) # Append width [wx, wy]

    # Calculate pairwise distance matrix between spots1 and spots2
    dm = distance_matrix(s1_pos,s2_pos)
    # Discard links where distance exceeds threshold
    dm[dm > params.colocalize_distance] = -1

    # Calculate Gaussian overlap factor for potential links
    overlap = np.zeros(dm.shape)
    for i in range(len(s1_pos)):
        for j in range(len(s2_pos)):
            # Only calculate overlap if distance is within threshold (dm >= 0)
            if dm[i,j] >= 0:
                # Calculate overlap based on distance and average widths
                # Formula assumes Gaussian spots
                width_sq_1 = (0.5*(s1_width[i][0][0]+s1_width[i][0][1]))**2 # Avg squared width spot 1
                width_sq_2 = (0.5*(s2_width[j][0][0]+s2_width[j][0][1]))**2 # Avg squared width spot 2
                overlap[i,j] = np.exp(-dm[i,j]**2 / (2. * (width_sq_1 + width_sq_2)))

    # Filter links based on minimum overlap threshold
    overlap[overlap < 0.75] = 0
    # Find indices (rows from s1, cols from s2) of valid overlaps
    indices = np.where(overlap!=0)

    # Extract trajectory IDs for the linked spots
    id1=[]; id2=[]
    for i in range(len(indices[0])):
        id1.append(spots1[indices[0][i]][2]) # Get traj ID from spots1 using row index
        id2.append(spots2[indices[1][i]][2]) # Get traj ID from spots2 using col index

    # Return arrays of linked trajectory IDs
    return np.array(id1), np.array(id2)


def straightline(x, m, c): # Simple linear function for curve fitting
    # Returns y = mx + c
    return m * x + c

def get_copy_number(params, calculated_isingle, channel=None): # Calculates copy number per cell from total intensity
    # Read image data
    image_data = images.ImageData()
    image_data.read(params) # Assumes params object has 'name' attribute for filename
    # Get the specific frame (using laser_on_frame - check where this is defined)
    # Might be better to average over several frames if possible
    frame = image_data.pixel_data[images.laser_on_frame,:,:] # 'spots' undefined here, assume images.laser_on_frame?

    # Initialise list for copy numbers
    copy_nums = []
    # Open output file
    f = open(params.name + "_copy_numbers.csv", "w") # Added "w" mode
    f.write("Cell\tCopy number\n") # Write header
    # Calculate background from masked area (pixels == 0)
    bg = np.mean(frame[image_data.mask_data==0]) # Assumes mask_data exists
    # Iterate through each cell mask (identified by integers > 0)
    for i in range(1,np.amax(image_data.mask_data)+1):
        # Calculate total background-subtracted intensity for the cell
        cell_intensity = np.sum(frame[image_data.mask_data==i] - bg)
        # Calculate copy number by dividing by single molecule intensity
        copy_nums.append(cell_intensity / calculated_isingle)
        # Write cell ID and calculated copy number to file
        f.write(str(i)+"\t"+str(copy_nums[i-1])+"\n")
    # Close output file
    f.close()

def plot_snr(params,snr,channel=None): # Plots histogram and KDE of Signal-to-Noise Ratios (SNR)
    # Define bandwidth for Kernel Density Estimation (KDE)
    bandwidth=0.07
    # Create KDE function from non-zero SNR values
    kde = gaussian_kde(snr[snr != 0], bw_method=bandwidth)
    # Create x-axis values for plotting KDE curve
    x = np.linspace(0, np.amax(snr), 10000)
    # Evaluate KDE at x values
    pdf = kde.evaluate(x)

    # --- Plotting ---
    fig, ax1 = plt.subplots()
    # Plot histogram of SNR values
    ax1.hist(snr[snr != 0], bins=np.arange(0,np.amax(snr)+2,0.05), label="Raw data")
    # Create second y-axis for KDE plot
    ax2 = ax1.twinx()
    # Plot KDE curve
    ax2.plot(x, pdf, 'k-', label="Gaussian KDE") # Corrected label
    # Format KDE y-axis in scientific notation
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    # Add legend
    plt.legend() # May only show KDE legend due to twin axes, consider combining legends

    # Set title and output filename based on channel
    if channel=="L":
        plt.title("Left hand channel SNR plot")
        outseed = params.name + "_Lchannel_SNR" # Underscore added
    elif channel=="R":
        plt.title("Right hand channel SNR plot")
        outseed = params.name + "_Rchannel_SNR" # Underscore added
    else:
        plt.title("Whole frame SNR plot")
        outseed = params.name + "_SNR"

    # Save the plot
    plt.savefig(outseed+"_plot.png", dpi=300)
    # Show plot if display enabled
    if params.display_figures:
        plt.show()
    plt.close()

    # Find the SNR value at the peak of the KDE
    peak = x[np.where(pdf == np.amax(pdf))]
    # Write raw SNR data to file
    ofile = params.name + "_snr_data.csv" # Changed filename to be specific
    f = open(ofile, 'w')
    f.write("SNR\n") # Add header
    for i in range(len(snr)):
        f.write(str(snr[i])+"\n")
    f.close()
    # Return the peak SNR value
    return peak


def get_isingle(params, intensities, channel=None): # Estimates single-molecule intensity (Isingle) using KDE
    # Define scaling factor for plot x-axis
    scale = 3
    # Filter out very low intensity values (likely noise)
    intensities = intensities[intensities > 1]
    # Define KDE bandwidth
    bandwidth = 0.1
    # Create KDE function
    kde = gaussian_kde(intensities, bw_method=bandwidth)
    # Create x-axis values for plotting (up to max intensity)
    x = np.linspace(0, np.amax(intensities), int(np.amax(intensities)))
    # Evaluate KDE
    pdf = kde.evaluate(x)
    # Find intensity value at the peak of the KDE (estimate of Isingle)
    peak = x[np.where(pdf == np.amax(pdf))].astype('int')
    # Get the height of the KDE peak
    peakval = np.amax(pdf)

    # --- Plotting ---
    fig, ax1 = plt.subplots()
    # Set plot labels
    plt.xlabel("Intensity (camera counts per pixel x$10^{}$)".format(scale)) # Corrected f-string
    plt.ylabel("Number of foci")
    # Plot histogram of intensities (scaled)
    l1 = ax1.hist(
        intensities/10**scale,
        bins=np.arange(0, np.amax(intensities/10**scale) + 100/10**scale, 100/10**scale),
        label="Raw data", color="gray"
    )
    # Create second y-axis for KDE plot
    ax2 = ax1.twinx()
    # Plot KDE curve (scaled x and y)
    l2 = ax2.plot(x/10**scale, pdf*10**scale, "k-", label="Gaussian KDE")
    plt.ylabel("Probability density (a.u.)")
    # Format KDE y-axis in scientific notation
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    # Plot vertical line at the estimated Isingle peak
    l3 = ax1.plot([peak/10**scale,peak/10**scale],[0,ax1.get_ylim()[1]], 'r--', label="Isingle", lw=2) # Use ax1 ylim
    
    # Set title and output filename based on channel
    if channel=="L":
        plt.title("Left hand channel intensity plot\nIsingle = {}".format(np.round(peak))) # Corrected f-string
        outseed = params.name + "_Lchannel_intensities" # Underscore added
    elif channel=="R":
        plt.title("Right hand channel intensities plot\nIsingle = {}".format(np.round(peak))) # Corrected f-string
        outseed = params.name + "_Rchannel_intensity" # Underscore added
    else:
        plt.title("Whole frame intensities\nIsingle = {}".format(np.round(peak))) # Corrected f-string, uncommented
        outseed = params.name + "_intensity"

    # Combine legends from both axes
    lns = l2+l3 # Add l1? lns = l1.get_label()+l2+l3? Needs testing.
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0) # Place legend automatically
    # Save plot
    plt.savefig(outseed+"_plot.png", dpi=300)
    # Show plot if display enabled
    if params.display_figures:
        plt.show()
    plt.close()
    # Write raw intensity data to file
    ofile = outseed + "_data.csv"
    f = open(ofile, 'w')
    f.write("Intensity\n") # Add header
    for i in range(len(intensities)):
        f.write(str(intensities[i])+"\n")
    f.close()
    # Return the estimated Isingle peak value
    return peak

def get_diffusion_coef(traj_list, params, channel=None): # Calculates Diffusion Coeff (D) and Loc. Precision (sigma) from MSD
    # Initialise lists to store results
    diffusion_coefs = []
    loc_precisions = []

    # --- Setup for plotting Mean Squared Displacement (MSD) ---
    plt.figure(figsize=(8, 6)) # Create figure for MSD plots
    msd_plot_count = 0 # Counter for plotting limited number of MSDs
    # --- End MSD plotting setup ---

    # Iterate through each trajectory in the list
    for traj in traj_list:
        # Get trajectory length
        trajectory_length = traj.length
        # Skip trajectories shorter than required points for MSD fit
        if trajectory_length < params.msd_num_points + 1:
            continue

        # Initialise arrays for MSD calculation
        MSD = np.zeros(trajectory_length - 1)  # Mean Squared Displacement values
        tau = np.zeros(trajectory_length - 1)  # Time lag values
        track_lengths = np.zeros(trajectory_length - 1) # Number of points used for each MSD lag
        # Get x, y coordinates in physical units (microns)
        x = np.array(traj.path)[:, 0] * params.pixel_size
        y = np.array(traj.path)[:, 1] * params.pixel_size

        # Calculate MSD for different time lags (i)
        for i in range(1, trajectory_length):
            # Optimisation: Stop calculation beyond max points needed for fit
            if i > params.msd_num_points:
                break
            # Calculate squared displacements for lag i
            sqd = (x[i:] - x[: trajectory_length - i]) ** 2 + (
                y[i:] - y[: trajectory_length - i]
            ) ** 2
            # Calculate mean if points exist
            if len(sqd) > 0:
                MSD[i - 1] = np.mean(sqd)
                track_lengths[i - 1] = len(sqd) # Store number of points averaged
                tau[i - 1] = i * params.frame_time # Store time lag
            else:
                # Handle case with no points (shouldn't happen with optimisation)
                MSD[i - 1] = np.nan
                track_lengths[i - 1] = 0
                tau[i-1] = i * params.frame_time

        # Select only the first 'msd_num_points' for fitting
        tau = tau[: params.msd_num_points]
        MSD = MSD[: params.msd_num_points]
        track_lengths = track_lengths[: params.msd_num_points]

        # Filter out any potential NaN values (e.g., from very short trajectories)
        valid_indices = ~np.isnan(MSD)
        tau = tau[valid_indices]
        MSD = MSD[valid_indices]
        track_lengths = track_lengths[valid_indices]

        # Check if enough points remain for fitting
        if len(tau) < 2: # Need at least 2 points for a line fit
             print(f"WARNING: Trajectory ID {traj.id} too short for fitting after processing.") # Added traj ID
             continue

        # Calculate weights for fitting (less weight for longer lags with fewer points)
        # Avoid division by zero if track_lengths are all zero
        weights = track_lengths.astype("float32") / float(
            np.amax(track_lengths)
        ) if np.amax(track_lengths) > 0 else np.ones_like(track_lengths)

        # --- Plot individual MSD curve (optional, limited number) ---
        if msd_plot_count < 10:
            plt.plot(tau, MSD, marker='o', linestyle='--', label=f'Traj {traj.id}') # Added traj ID
            msd_plot_count += 1
        # --- End plotting ---

        try:
            # Fit a straight line (y = mx + c) to MSD vs tau using weighted least squares
            # sigma = 1.0/weights assumes weights represent relative std dev (larger weight = smaller error)
            # absolute_sigma=False means errors are relative
            popt, pcov = curve_fit(straightline, tau, MSD, p0=[1, 0], sigma=1.0/weights, absolute_sigma=False)

            # --- Process Fit Results ---
            # Check if slope (popt[0]) is physically meaningful (non-negative)
            if popt[0] >= 0:
                # Calculate Diffusion Coefficient (D = slope / 4 for 2D)
                diffusion_coef = popt[0] / 4.0
                diffusion_coefs.append(diffusion_coef)

                # Calculate Localisation Precision (sigma) if intercept is positive
                if popt[1] > 0:
                    # Formula: MSD(0) = 4*sigma^2 => sigma = sqrt(intercept)/2
                    loc_prec = np.sqrt(popt[1]) / 2.0
                    loc_precisions.append(loc_prec)

                # --- Plot fitted line (optional) ---
                if msd_plot_count <= 10: # Match MSD plot condition
                    plt.plot(tau, straightline(tau, *popt), color=plt.gca().lines[-1].get_color(), linestyle='-')
                # --- End plotting ---

            else:
                # Warn if slope is negative
                print(f"WARNING: Fit resulted in negative slope ({popt[0]:.2e}) for trajectory {traj.id}. Skipping.") # Added traj ID
            # --- End processing fit results ---

        except Exception as e: # Catch any other fitting errors
            print(f"WARNING: Unable to fit curve for trajectory {traj.id}. Error: {e}") # Added traj ID

    # --- Finalise MSD Plot ---
    plt.xlabel(r"Time Lag $\tau$ (s)")
    plt.ylabel("MSD ($\mu$m$^2$)")
    plt.title(f"MSD vs Time Lag (First {msd_plot_count} trajectories)")
    if msd_plot_count > 0: plt.legend() # Add legend if plots were made
    if params.display_figures: plt.show() # Show plot if enabled
    plt.close() # Close the MSD figure
    # --- End MSD Plot Finalisation ---

    # --- Plot Histogram of Diffusion Coefficients ---
    plt.figure(figsize=(8, 6)) # Create new figure
    if diffusion_coefs: # Check if any D values were calculated
        plt.hist(diffusion_coefs, bins='auto') # Plot histogram with auto bins
        mean_diff_coeff = np.mean(diffusion_coefs)
        # Create title with mean D and count
        hist_title = f"Diffusion Coefficients\nMean = {mean_diff_coeff:.2f} $\mu$m$^2$s$^{-1}$ (n={len(diffusion_coefs)})"
    else:
        # Display message if no valid D values found
        plt.text(0.5, 0.5, "No valid diffusion coefficients found", horizontalalignment='center', verticalalignment='center')
        mean_diff_coeff = np.nan
        hist_title = "Diffusion Coefficients (No valid data)"

    plt.xlabel("Diffusion coefficient ($\mu$m$^2$s$^{-1}$)")
    plt.ylabel("Number of Trajectories")

    # Set title and output filename based on channel
    if channel=="L":
        plt.title(f"Left channel {hist_title}")
        ofile = params.name + "_Lchannel_diff_coeff.png"
    elif channel=="R":
        plt.title(f"Right channel {hist_title}")
        ofile = params.name + "_Rchannel_diff_coeff.png"
    else:
        plt.title(hist_title)
        ofile = params.name + "_diff_coeff.png"

    # Save histogram plot
    print(f"Saving histogram to: {ofile}")
    plt.savefig(ofile, dpi=300, bbox_inches='tight') # Use tight layout
    if params.display_figures: plt.show() # Show plot if enabled
    plt.close() # Close the histogram figure
    # --- End Histogram Plot ---

    # --- Save Data to Files ---
    # Save Diffusion Coefficients
    diff_coeff_file = params.name + "_diff_coeff_data.csv"
    print(f"Saving diffusion coefficients to: {diff_coeff_file}")
    with open(diff_coeff_file, "w") as f:
        f.write("DiffusionCoefficient_um2_s\n") # Header
        for dc in diffusion_coefs:
            f.write(f"{dc:.6f}\n") # Formatted output

    # Save Localization Precisions
    loc_prec_file = params.name + "_diff_coeff_loc_precision_data.csv"
    print(f"Saving localization precisions to: {loc_prec_file}")
    with open(loc_prec_file, "w") as f:
        f.write("LocalizationPrecision_um\n") # Header
        for lp in loc_precisions:
            f.write(f"{lp:.6f}\n") # Formatted output
    # --- End Save Data ---

    # Return calculated D and sigma values
    return diffusion_coefs, loc_precisions


def plot_traj_intensities(params, trajs, channel=None, chung_kennedy=True): # Plots intensity traces for trajectories
    # Initialise list for Chung-Kennedy filtered data if enabled
    if chung_kennedy: ck_data = []
    # Plot raw intensity traces for each trajectory
    for traj in trajs:
        t = np.array(traj.intensity) # Get intensity array
        plt.plot(t/10**3) # Plot scaled intensity vs frame index
        # Apply Chung-Kennedy filter (currently commented out)
        #ck_data.append(chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1])

    # --- Save Chung-Kennedy Data (if calculated) ---
    # Note: ck_data is currently always empty as filtering is commented out
    ofile = params.name+"_chung_kennedy_data.csv"
    f = open(ofile, 'w')
    # The following block is problematic if ck_data traces have different lengths
    # ck_data = np.array(ck_data) ##DWA## shapes not same, removed conversion
    for ck_trace in ck_data: # Corrected loop variable
        # Write each trace as comma-separated values
        f.write(str(ck_trace[0])) # Write first element
        for j in range(1, len(ck_trace)): # Write remaining elements
            f.write(","+str(ck_trace[j]))
        f.write("\n")
    f.close()
    # --- End Save CK Data ---

    # --- Format and Save Raw Intensity Plot ---
    plt.xlabel("Frame number")
    plt.ylabel("Intensity (camera counts per pixel x$10^3$)")
    # Set title and output filename based on channel
    if channel=="L":
        plt.title("Left channel trajectory intensity")
        ofile = params.name+"_Lchannel_trajectory_intensities.png"
    elif channel=="R":
        plt.title("Right channel trajectory intensity")
        ofile = params.name+"_Rchannel_trajectory_intensities.png"
    else:
        plt.title("Whole frame trajectory intensity")
        ofile = params.name+"_trajectory_intensities.png"
    # Save the plot
    plt.savefig(ofile, dpi=300)
    # Show plot if enabled
    if params.display_figures:
        plt.show()
    plt.close()
    # --- End Raw Intensity Plot ---

    # --- Plot Chung-Kennedy Filtered Data (if calculated) ---
    if chung_kennedy and ck_data: # Only plot if enabled AND data exists
        # Plot filtered traces
        for ck in ck_data:
            plt.plot(ck)
        # Formatting
        plt.xlabel("Frame number")
        plt.ylabel("Intensity (camera counts per pixel x$10^3$)") # Should units change after CK?
        # Set title and output filename based on channel
        if channel=="L":
            plt.title("Left channel Chung-Kennedy intensity")
            ofile = params.name+"_Lchannel_CK_filtered_intensities.png"
        elif channel=="R":
            plt.title("Right channel Chung-Kennedy intensity")
            ofile = params.name+"_Rchannel_CK_filtered_intensities.png"
        else:
            plt.title("Whole frame Chung-Kennedy intensity")
            ofile = params.name+"_CK_filtered_intensities.png"
        # Format y-axis
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
        # Save plot
        plt.savefig(ofile, dpi=300)
        # Show plot if enabled
        if params.display_figures:
            plt.show()
        plt.close()
    # --- End CK Plot ---

def get_stoichiometries(trajs, isingle, params, channel=None): # Calculates stoichiometry for each trajectory
    # Initialise list to store stoichiometries
    stoics = []
    # Find the earliest start frame among trajectories long enough for analysis
    startframe = 100000 # Initialise with a large number
    for traj in trajs:
        if traj.start_frame < startframe and traj.length >= params.num_stoic_frames:
            startframe = traj.start_frame

    # Iterate through trajectories to calculate stoichiometry
    for traj in trajs:
        # Skip trajectories shorter than the required number of frames
        if traj.length < params.num_stoic_frames:
            continue

        # Calculate stoichiometry based on the chosen method
        if params.stoic_method == "Initial":
            # Use intensity of the very first frame
            traj.stoichiometry = traj.intensity[0] / isingle
            # Ensure it's a scalar value (isingle might be array)
            traj.stoichiometry = traj.stoichiometry[0] if isinstance(traj.stoichiometry, np.ndarray) else traj.stoichiometry
        elif params.stoic_method == "Mean":
            # Use mean intensity of the first N frames
            traj.stoichiometry = (
                np.mean(traj.intensity[: params.num_stoic_frames]) / isingle
                )
            # Ensure scalar if needed
            traj.stoichiometry = traj.stoichiometry[0] if isinstance(traj.stoichiometry, np.ndarray) else traj.stoichiometry
        elif params.stoic_method == "Linear":
            # Skip if trajectory starts too late relative to earliest start (threshold = 4 frames)
            if traj.start_frame - startframe > 4:
                continue
            else:
                # Prepare data for linear fit (frame index vs intensity)
                xdata = np.arange(0, params.num_stoic_frames , dtype="float")
                ydata = traj.intensity[0: params.num_stoic_frames]
                try: # Add error handling for curve_fit
                    # Fit straight line y = mx + c
                    popt, pcov = curve_fit(straightline, xdata, ydata)
                    intercept = popt[1] # Get the intercept (intensity at frame 0 of trace)
                    slope = popt[0] # Get the slope

                    # Calculate stoichiometry using intercept, adjusted for start frame delay
                    # Condition 'intercept > 0' ensures physical sense
                    if intercept > 0:
                         # Extrapolate back to the *absolute* frame 0 using the slope and start frame difference
                         # abs((traj.start_frame-startframe)*slope) estimates intensity lost before tracking started
                        stoichiometry_val = (intercept + abs((traj.start_frame-startframe)*slope)) / isingle
                        # Ensure scalar
                        traj.stoichiometry = stoichiometry_val[0] if isinstance(stoichiometry_val, np.ndarray) else stoichiometry_val
                    else:
                         # Skip if intercept is not positive
                        continue
                except RuntimeError: # Handle fit failure
                    print(f"Warning: Linear fit failed for trajectory {traj.id}. Skipping stoichiometry calculation.")
                    continue
        else: # Handle invalid method
             print(f"Warning: Invalid stoic_method '{params.stoic_method}'. Skipping stoichiometry calculation for trajectory {traj.id}.")
             continue

        # Append the calculated stoichiometry (if successful) to the list
        # Check if traj.stoichiometry was actually assigned
        if hasattr(traj, 'stoichiometry'):
            stoics.append(traj.stoichiometry)

    # --- Plotting and Saving Results ---
    # Check if enough stoichiometry values were calculated for plotting/KDE
    if len(stoics) < 2:
        print(f"Warning: Not enough stoichiometry values (found {len(stoics)}) to create KDE and histograms.")
        # Create empty output file if no valid data
        if len(stoics) == 0:
            print("No valid stoichiometries found. Check filtering criteria.")
            # Define output file seed based on channel
            if channel=="L": oseed = params.name+"_Lchannel_stoichiometry"
            elif channel=="R": oseed = params.name+"_Rchannel_stoichiometry"
            else: oseed = params.name+"_stoichiometry"
            # Write empty file
            with open(oseed + "_data.csv", "w") as f:
                f.write("# No valid stoichiometries found\n")
            return 0 # Exit function

    # Convert list to numpy array for easier handling
    stoics = np.array(stoics)

    # Determine maximum stoichiometry for plot limits
    if len(stoics) >= 1:
        max_stoic = int(np.round(np.amax(stoics))) if np.amax(stoics) > 0 else 1 # Handle case where max is 0 or less
    else:
        max_stoic = 1 # Fallback

    # --- Histogram Plot ---
    fig, ax1 = plt.subplots()
    # Plot histogram if data exists
    if len(stoics) > 0:
        l1 = ax1.hist(
            stoics,
            bins=np.arange(0, max_stoic + 2, 1), # Bins from 0 to max_stoic+1
            color="gray"
        )
    else:
        l1 = ax1.bar([], [], color="gray") # Empty plot

    # Add KDE plot if enough data points exist
    if len(stoics) >= 2:
        bandwidth = 0.7 # KDE bandwidth
        kde = gaussian_kde(stoics, bw_method=bandwidth)
        x = np.linspace(0, max_stoic + 1, max(1, max_stoic + 1)*10) # X values for KDE curve
        pdf = kde.evaluate(x)
        # Plot KDE on second y-axis
        ax2 = ax1.twinx()
        l2 = ax2.plot(x, pdf, "k-", label="Gaussian KDE")
        ax2.set_ylabel("Probability density (a.u.)") # Corrected label
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))

    # Set axis labels and ticks
    plt.xlabel("Rounded stoichiometry")
    ax1.set_ylabel("N") # Label for histogram counts
    if len(stoics) > 0:
        plt.xticks(range(0, max_stoic+2)) # Ticks from 0 to max_stoic+1

    # Set title and output filename seed
    if channel=="L":
        plt.title("Left channel stoichiometry")
        oseed = params.name+"_Lchannel_stoichiometry"
    elif channel=="R":
        plt.title("Right channel stoichiometry")
        oseed = params.name+"_Rchannel_stoichiometry"
    else:
        plt.title("Whole frame stoichiometry")
        oseed = params.name+"_stoichiometry"

    # Save histogram plot
    plt.savefig(oseed+"_histogram.png", dpi=300)
    if params.display_figures: plt.show()
    plt.close()
    # --- End Histogram Plot ---

    # --- Scatter Plot ---
    if len(stoics) > 0:
        plt.scatter(range(len(stoics)), stoics) # Plot stoichiometry vs spot index
        plt.xlabel("Spot #")
        plt.ylabel("Raw stoichiometry")
        # Set title based on channel
        if channel=="L": plt.title("Left channel stoichiometry")
        elif channel=="R": plt.title("Right channel stoichiometry")
        else: plt.title("Whole frame stoichiometry")
        # Save scatter plot
        plt.savefig(oseed+"_scatter.png", dpi=300)
        if params.display_figures: plt.show()
        plt.close()
    # --- End Scatter Plot ---

    # --- Save Data ---
    with open(oseed + "_data.csv", "w") as f:
        f.write("Stoichiometry\n") # Header
        for stoic_val in stoics: # Corrected loop variable
            f.write(f"{float(stoic_val)}\n") # Write each value
    # --- End Save Data ---

    return 0 # Indicate successful completion