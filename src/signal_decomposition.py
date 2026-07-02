# signal_decomposition.py
''' 
This script will explain and implement signal decomposition of four ground stations along the 
cascades region.
Jun 26, 2026
Version 1
Chelsea Momoh
'''

#Step 1: Mathematical Reasoning.

'''
Our data consists of three vectors I will name y_north, y_east, and y_vertical. 
They are the displacement in the north, east, and vertical directions at each epoch.

I found/assume each displacement, y, can be approximated using 
    - tectonic velocity
    - annual amplitude and phase (wiggle caused by snow melt/groundwater)
    - semiannual amplitude and phase (wiggle caused by other physcial processes)

NOTE: In the future, I'd like to perform analyses to determine whether these are the best parameters for all 6 stations.
In the meantime, I make this assumption to move forward with decomposition.

For example, for a single epoch, we represent a displacement in the east direction as such:

y_east_i = a + bt_i + csin(2PIt_i) + dcos(2PIt_i) + esin(4PIt_i) + fcos(4PIt_i) , 

Where 
a -> value of intercept @ t = 0
b -> linear coefficient of tectonic velocity (tectonic velocuty is constant)
c & d -> used in combination as linear coefficients to weight the amplitude and phase of the annual cycle
e & f -> used in combination as linear coefficients to weight the amplitude and phase of the SEMI-annual cycle
t -> decimal years since reference point (this is the time)
y -> millimiters of displacement in the east direction at the ith epoch


We don't represent the quarterly cycle to avoid overfitting. Also because quarterly cycles don't have significant physical processes.

So we'ev modeled a single epoch in the east direction in the equation above, but there are thousands of lines affecting abcde&f, and we can't solve for that 5000+ times for each line in the dataset.
So instead, we solve all at once using a matrix formula.

XB_east = Y_east

where 
X = [1 t_1 sin(2PIt_1) cos(2PIt_1) sin(4PIt_1) cos(4PIt_1)
     1 t_2 sin(2PIt_2) cos(2PIt_2) sin(4PIt_2) cos(4PIt_2)
     1 t_3 sin(2PIt_3) cos(2PIt_3) sin(4PIt_3) cos(4pPIt_3)
     . . .
     .
     .
     1 t_n sin(2PIt_n) cos(2PIt_n) sin(4PIt_n) cos(4PIt_n) ]

B = transpose([a b c d e f]) for the east direction

So X is the same for all three directions. 
But Y_north, Y_east, and Y_vertical change, so we'll need different coefficients. In other words, 
different B_east, B_vertical, B_north.

Got me?
'''


#Step 2: Import libraries and load my data from the json files
import os
os.chdir("/workspaces/GNSS/data")
import pandas as pd
import json
import numpy as np
#PSD imports
from scipy.signal import periodogram
import matplotlib.pyplot as plt

with open("metadata.json", "r") as f:
    metadata = json.load(f) #I dont actually use this in this script.


p349 = pd.read_json("p349.json", orient="records")
p349['Date'] = pd.to_datetime(p349['Date'])

p380 = pd.read_json("p380.json", orient="records")
p380['Date'] = pd.to_datetime(p380['Date'])

p434 = pd.read_json("p434.json", orient="records")
p434['Date'] = pd.to_datetime(p434['Date'])

p441 = pd.read_json("p441.json", orient="records")
p441['Date'] = pd.to_datetime(p441['Date'])



#step 3: Fit the design models

#@Brief: This section will utilize datetime to perform datetime arithmetic and find t. 
#The specific dataset I use is not arbitrary. Each station dataset has a different number of elapsed days, so i have to repeat this process four times.
# The formula to convert the number of days to decimal years and get time deltas (changes in time) is given as:

elapsed = p349['Date'] - p349['Date'].iloc[0] 
t_p349 = elapsed.dt.days / 365.25

elapsed = p380['Date'] - p380['Date'].iloc[0] 
t_p380 = elapsed.dt.days / 365.25

elapsed = p434['Date'] - p434['Date'].iloc[0] 
t_p434 = elapsed.dt.days / 365.25

elapsed = p441['Date'] - p441['Date'].iloc[0] 
t_p441 = elapsed.dt.days / 365.25


#@Brief: This section will construct the design matrices for X for all four stations.

X_p349 = np.column_stack([
    np.ones(len(t_p349)),  # ones
    t_p349,  # t
    np.sin(2 * np.pi * t_p349),  # sin(2π t)
    np.cos(2 * np.pi * t_p349),  # cos(2π t)
    np.sin(4 * np.pi * t_p349),  # sin(4π t)
    np.cos(4 * np.pi * t_p349)   # cos(4π t)
])

X_p380 = np.column_stack([
    np.ones(len(t_p380)),  # ones
    t_p380,  # t
    #I must use np.sin for element-wise operations. Numpy is great for element-wise stuffs!
    np.sin(2 * np.pi * t_p380),  # sin(2π t)
    np.cos(2 * np.pi * t_p380),  # cos(2π t)
    np.sin(4 * np.pi * t_p380),  # sin(4π t)
    np.cos(4 * np.pi * t_p380)   # cos(4π t)
])

X_p434 = np.column_stack([
    np.ones(len(t_p434)),  # ones
    t_p434,  # t
    np.sin(2 * np.pi * t_p434),  # sin(2π t)
    np.cos(2 * np.pi * t_p434),  # cos(2π t)
    np.sin(4 * np.pi * t_p434),  # sin(4π t)
    np.cos(4 * np.pi * t_p434)   # cos(4π t)
])

X_p441 = np.column_stack([
    np.ones(len(t_p441)),  # ones
    t_p441,  # t
    np.sin(2 * np.pi * t_p441),  # sin(2π t)
    np.cos(2 * np.pi * t_p441),  # cos(2π t)
    np.sin(4 * np.pi * t_p441),  # sin(4π t)
    np.cos(4 * np.pi * t_p441)   # cos(4π t)
])

#@Brief: This section will extract the y vectors from the dataframes and perform 
# least squares approx to solve for B for all four stations.

#This gives me the y vectors for each direction for each station 
y_north_p349 = p349['North (mm)'].values
y_east_p349 = p349['East (mm)'].values
y_vert_p349 = p349['Vertical (mm)'].values

y_north_p380 = p380['North (mm)'].values
y_east_p380 = p380['East (mm)'].values
y_vert_p380 = p380['Vertical (mm)'].values

y_north_p434 = p434['North (mm)'].values
y_east_p434 = p434['East (mm)'].values
y_vert_p434 = p434['Vertical (mm)'].values

y_north_p441 = p441['North (mm)'].values
y_east_p441 = p441['East (mm)'].values
y_vert_p441 = p441['Vertical (mm)'].values


#@Brief: this next section gives me the coefficients for each direction for all four stations.
#The underscores get rid of unneeeded data (e.g., the residuals provided here are not a time series, but a summary stat, not helpful.)
beta_north_p349, _, _, _ = np.linalg.lstsq(X_p349, y_north_p349, rcond=None) 
beta_east_p349, _, _, _ = np.linalg.lstsq(X_p349, y_east_p349, rcond=None) 
beta_vert_p349, _, _, _ = np.linalg.lstsq(X_p349, y_vert_p349, rcond=None) 

beta_north_p380, _, _, _ = np.linalg.lstsq(X_p380, y_north_p380, rcond=None) 
beta_east_p380, _, _, _ = np.linalg.lstsq(X_p380, y_east_p380, rcond=None) 
beta_vert_p380, _, _, _ = np.linalg.lstsq(X_p380, y_vert_p380, rcond=None) 

beta_north_p434, _, _, _ = np.linalg.lstsq(X_p434, y_north_p434, rcond=None) 
beta_east_p434, _, _, _ = np.linalg.lstsq(X_p434, y_east_p434, rcond=None) 
beta_vert_p434, _, _, _ = np.linalg.lstsq(X_p434, y_vert_p434, rcond=None) 

beta_north_p441, _, _, _ = np.linalg.lstsq(X_p441, y_north_p441, rcond=None) 
beta_east_p441, _, _, _ = np.linalg.lstsq(X_p441, y_east_p441, rcond=None) 
beta_vert_p441, _, _, _ = np.linalg.lstsq(X_p441, y_vert_p441, rcond=None) 

'''
I wrote print(beta_north_p349) and got
[-3.33226909  6.91818185  0.03229881 -0.7911755   0.26515507 -0.06557577]
Notice a = -3.33 despite the data showing a displacement of zero at this point. This is just noise around the intercept!
'''

print("Completed building design matrices.")

#@Brief: This section will find the residuals for each groundstation's directions.
#Residuals are milimeters of variation the model did not explain.
#residuals=y−Xβ

resid_p349_north = y_north_p349 - X_p349@beta_north_p349
resid_p349_east = y_east_p349 - X_p349@beta_east_p349
resid_p349_vert = y_vert_p349 - X_p349@beta_vert_p349

resid_p380_north = y_north_p380 - X_p380@beta_north_p380
resid_p380_east = y_east_p380 - X_p380@beta_east_p380
resid_p380_vert = y_vert_p380 - X_p380@beta_vert_p380

resid_p434_north = y_north_p434 - X_p434@beta_north_p434
resid_p434_east = y_east_p434 - X_p434@beta_east_p434
resid_p434_vert = y_vert_p434 - X_p434@beta_vert_p434

resid_p441_north = y_north_p441 - X_p441@beta_north_p441
resid_p441_east = y_east_p441 - X_p441@beta_east_p441
resid_p441_vert = y_vert_p441 - X_p441@beta_vert_p441

print(f"Finished building residual series.")

#step 4: Characterization of noise reasoning 

#@Brief: Explaining noise characterization 

'''
My residual series (my residual vectors) are all time series. 
(In the Fourier sense, they are the sum of sinusoids.)
Any pattern that repeats itself in this time series can be described by its frequency. 
e.g.
A pattern repeating every 10 years has 0.1 cycles per year.
A patter repeating every 6 months has 2 cycles per year. 
The lowest possible frequency for my data would have to happen every 20 years. Or one cycle over the course of 20 years.
The Highest frequency would equal the sampling rate, so it would happen each day. One cycle each 2 days.

Power Spectral Density (PSD) tells, for every frequency of sinusoid, how much power (variance) the signal
contains at that frequency. PSD is a decomposition of the signal across different frequencies (patterns).


The plot of the PSD on a loglog plot (loglog for easier fitting) reveals the type of noise I'm dealing with.

Typically, when we estimate, for example, the velocity coefficient, the uncertainty of the 
estimate depends on theresidual noise. Standard least squares assumes white (random/normal/gaussian) noise.
With white noise, the more data one acquires, the more accurate a prediction becomes. 

However, with colored (Pink/Flisker or Random Walk) noise, noise is not independent. 
Each measuremnt is affected by the last. Physically, we attribute this noise to unmodeled 
physical processes.

By plotting the PSD's shape, I can determine the type of noise I am working with, construct the 
proper covariance matrix, and plug that matrix into an appropriate uncertainty formula to get
a MORE REALISTIC ESTIMATE OF THE ERROR IN OUR VELOCITY (trend) AND ANNUAL/SEMI-ANUAL PROCESSES.

Otherwise, we'd have a pretty bad idea of how trashy our measurements are.

*NOTE: We are not tracking the large signals of the North American tectonic plate's movement. We
are tracking the deviation of stations' movement from the plate due to seasonal loading signals, post-earthquake
deformation, etc. Recall NAM14 (the data I downloaded, see data folder) removes plate movement signals. 
This essentially 'centers data around the mean,' preventing large tectonic movement from hiding smaller signals. 

Ya feel?  

'''

#@Brief: Scipy Periodogram
#Periodogram takes my residual array and sampling frequency to output frequencies and power (variances)
#I have one sample per day, and we want to define our frequency in cycles per year. 
# let f = 365.25 (1/4 for leap years)

'''
freqs, power = periodogram(resid_p349_north, fs=365.25)
I did print(freqs[:10]) and got:
[0. 0.04854466 0.09708931 0.14563397 0.19417863 0.24272329 0.29126794 0.3398126  0.38835726 0.43690191]
This array lists the frequencies of the first ten found patterns. It lists each frequency in order. 
The longest frequency, as we can see, has 0.049 cycles per year, which is apparently ~20.6 yrs.
'''


"""
I did print(power[:10]) and got:
[2.78104284e-31 6.19735966e+01 2.85698624e+01 4.04845588e+01
 2.65258065e+00 3.20378456e+00 1.05712033e+00 4.64806175e-01
 5.13599883e-01 1.69800855e+00]
This array lists the variances of each pattern (the variances of each pattern) in the same order as pattern length.
We see very high power for the first few frequencies with a large drop off near higher frequencies.
Since the power is not relatively consistent throughout the array, we know the noise is not gaussion/random, 
but instead colored. 

*NOTE: for the frequency array, the first number is zero, suggesting a flat-line or no pattern. A constant.
However, our modeled accounted for any constant term with the intercept column of X, so the periodogram will 
pick up no variance for frequencies at 0. That's why the first (freq, power) term is (0, 2.78104284e-31) (basically 0,0)
"""


#step 5: Characterize residual noise using Power Spectral Density (PSD) plots

residuals = {
    "p349_north": resid_p349_north,
    "p349_east": resid_p349_east,
    "p349_vert": resid_p349_vert,

    "p380_north": resid_p380_north,
    "p380_east": resid_p380_east,
    "p380_vert": resid_p380_vert,

    "p434_north": resid_p434_north,
    "p434_east": resid_p434_east,
    "p434_vert": resid_p434_vert,

    "p441_north": resid_p441_north,
    "p441_east": resid_p441_east,
    "p441_vert": resid_p441_vert,
}

PSD_set = {}

for key, value in residuals.items(): #A mistake I make: must use residuals.items() not just the dict name
    freqs, power = periodogram(value, fs=365.25)
    PSD_set[key] = (freqs, power) #another mistake i make: freqs, power should be a tuple to allow for simple unpacking later on.


print("Plotting loglog PSD plots.")
'''
#I'm only commenting this out so I don't continue to create more plots each time I run!
for key, (freqs, power) in PSD_set.items():
    plt.figure()
    plt.loglog(freqs[1:], power[1:]) #We cannot plot the (0,0 pair)
    plt.xlabel("Frequency (cycles/year)")
    plt.ylabel("Power")
    plt.title(f"PSD — {key}")
    plt.tight_layout()
    plt.savefig(str(key)+".png", dpi=120)'''


#@Brief: In this section I will be printing out many many plots.
# Let's see if the power difference between the two frequency halves is truly downward then flat (colored then white).
#I will use bins to stabilize the trend line.
from scipy.stats import binned_statistic

def bin_psd(freqs, power, n_bins=30):
    # skip the zero-frequency point
    freqs = freqs[1:]
    power = power[1:]
    
    log_bins = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), n_bins)
    bin_means, bin_edges, _ = binned_statistic(freqs, power, statistic='mean', bins=log_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, bin_means

'''
#Commenting this out to avoid further images!
freqs, power = periodogram(residuals['p349_north'], fs=365.25)
bin_centers, bin_means = bin_psd(freqs, power)

plt.figure()
plt.loglog(freqs[1:], power[1:], alpha=0.3, label="raw")
plt.loglog(bin_centers, bin_means, color='red', linewidth=2, label="binned mean")
plt.xlabel("Frequency (cycles/year)")
plt.ylabel("Power")
plt.title("PSD — p349_north (binned)")
plt.legend()
plt.tight_layout()
plt.savefig("p349_north_binned.png", dpi=120)'''
#Looking at the red line, I can clearly see that there is a flattening happening at around 10^(0.7). 
# The flat section represent white noise, while the downward sloping section is colored. Maybe pink.

'''NOTE: This shorter record for P441 has generally gaussian noise throughout the dataset. 
         There is not enough of a clear downward slope for polyfit to catch colored noise

#commenting out a plot again
freqs, power = periodogram(residuals['p441_east'], fs=365.25)
bin_centers, bin_means = bin_psd(freqs, power)

plt.figure()
plt.loglog(freqs[1:], power[1:], alpha=0.3, label="raw")
plt.loglog(bin_centers, bin_means, color='red', linewidth=2, marker='o', label="binned mean")
plt.axvline(5, color='gray', linestyle='--', label="cutoff (5 cyc/yr)")
plt.xlabel("Frequency (cycles/year)")
plt.ylabel("Power")
plt.title("PSD — p441_east (binned)")
plt.legend()
plt.tight_layout()
plt.savefig("p441_east_binned.png", dpi=120)'''


'''
Finishing off the rest of the plots.
already_plotted = {"p349_north", "p441_east"}
 
for key, value in residuals.items():
    if key in already_plotted:
        continue
 
    freqs, power = periodogram(value, fs=365.25)
    bin_centers, bin_means = bin_psd(freqs, power)
 
    plt.figure()
    plt.loglog(freqs[1:], power[1:], alpha=0.3, label="raw")
    plt.loglog(bin_centers, bin_means, color='red', linewidth=2, marker='o', label="binned mean")
    plt.axvline(5, color='gray', linestyle='--', label="cutoff (5 cyc/yr)")
    plt.xlabel("Frequency (cycles/year)")
    plt.ylabel("Power")
    plt.title(f"PSD — {key} (binned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{key}_binned.png", dpi=120)
    plt.close()
 
print("Done plotting remaining PSDs.")'''





#NOTE:Let's set a line of demarcation at 10^(0.7), aorund 5 cycles/year

#@Brief: This section will fit the binned data for frequencies below my cutoff (for colored noise)
#For context, log(PSD)=log(A)−αlog(f). Recall we are plotting (frew, power) pairs. This function is in
#y = mx + b format, with each term having the log applied.
# α is the noise component/slope. log(f) is the input. log(A) is the intercept. log(PSD) is power.

#NOTE: CHARACTERIZE NOISE TYPES AS FOLLOWS: white noise (α≈0), flicker (α≈1), & random walk (α≈2)
#I need to find α for each direction for each station.

#np.polyfit(x, y, 1) fits a line to the data and returns [slope, intercept]

alphas = {}

print("alpha for for frequencies less than 10^(0.7) are as follows:")
for key, value in residuals.items():
    freqs, power = periodogram(value, fs=365.25)
    bin_centers, bin_means = bin_psd(freqs, power)
    mask = (bin_centers < 5) & (~np.isnan(bin_means)) #Prior mistake I made: ensure each event in the & statement is in parenthesis
    fit_freqs = bin_centers[mask]
    fit_power = bin_means[mask]
    log_f = np.log10(fit_freqs)
    log_p = np.log10(fit_power)
    slope, intercept = np.polyfit(log_f, log_p, 1)
    alpha = -slope
    print(f"{key}: {alpha}")
    alphas[key] = alpha

#@Brief: Persistent storage of alphas, Betas, residuals and X matrices

with open("alphas.json", "w") as f:
    json.dump(alphas, f, indent=2)

with open("residuals.json", "w") as f:
    #I have to use .tolist since numpy arrays can't be converted to JSON directly 
    json.dump({k: v.tolist() for k, v in residuals.items()}, f, indent=2)

betas = {
    "p349_north": beta_north_p349, "p349_east": beta_east_p349, "p349_vert": beta_vert_p349,
    "p380_north": beta_north_p380, "p380_east": beta_east_p380, "p380_vert": beta_vert_p380,
    "p434_north": beta_north_p434, "p434_east": beta_east_p434, "p434_vert": beta_vert_p434,
    "p441_north": beta_north_p441, "p441_east": beta_east_p441, "p441_vert": beta_vert_p441,
}
with open("betas.json", "w") as f:
    json.dump({k: v.tolist() for k, v in betas.items()}, f, indent=2)

X_matrices = {"p349": X_p349, "p380": X_p380, "p434": X_p434, "p441": X_p441}
with open("X_matrices.json", "w") as f:
    json.dump({k: v.tolist() for k, v in X_matrices.items()}, f, indent=2)

print("Completed persistent storage of alphas, betas, residuals, and X matrices. Prepped for outlier detection.")