# outlier_detection.py
''' 
This script will implement IQR, Z-score flagging, and a change-point detection model (PELT).
Jun 30, 2026
Version 1
Chelsea Momoh
'''

#Step 1: Load libraries and data
import os
import time
os.chdir("/workspaces/GNSS/data")
import pandas as pd
import numpy as np  
import json
#For PELT change point detection
import ruptures as rpt

with open("alphas.json", "r") as f:
    alphas = json.load(f)
with open("betas.json", "r") as f:
    betas = json.load(f)
betas = {key: np.array(value) for key, value in betas.items()}
with open("residuals.json", "r") as f:
    residuals = json.load(f)
residuals = {key: np.array(value) for key, value in residuals.items()}
with open("X_matrices.json", "r") as f:
    X_matrices = json.load(f)
X_matrices = {key: np.array(value) for key, value in X_matrices.items()}
with open("metadata.json", "r") as f:
    metadata = json.load(f)


p349 = pd.read_json("p349.json", orient="records")
p349['Date'] = pd.to_datetime(p349['Date'])
p380 = pd.read_json("p380.json", orient="records")
p380['Date'] = pd.to_datetime(p380['Date'])
p434 = pd.read_json("p434.json", orient="records")
p434['Date'] = pd.to_datetime(p434['Date'])
p441 = pd.read_json("p441.json", orient="records")
p441['Date'] = pd.to_datetime(p441['Date'])

stations = {'p349': p349, 'p380': p380, 'p434': p434, 'p441': p441}
station_dates = {
    "p349": p349['Date'], "p380": p380['Date'], "p434": p434['Date'], "p441": p441['Date'],
}
#Step 2: Outlier Detection Reasoning

'''
My residuals, the difference between my model and actual data, shoud be explained as noise, whether white or colored. 
However, outliers will occur under a set of circumstances, and these events should be flagged:
- Equipment changes/malfunctions, earthquakes, atmospheric noise, and glitches could afffect daa quality and lead to an outlier.
This script will utilize two tools, IQR/Z-score flaggin and change-point detection (PELT) to flag outliers.

Source: https://pipiras.sites.oasis.unc.edu/
Outlier: "have only an instantaneous effect" on data quality
Change points: "effect decays over time.... sustained for the entire series (or an extended portion of the series)"
//end source

Causes of outliers include atmospheric interruption or known noisy GNSS signals
Causes of change points include equipment changes (antenna or receiver swapped out) or earthquakes
--> The velocity of the signal after an earthquake changes for a period of time after the earthquake until it decays to baseline 

IQR/Z-score flagging catches outliers while PELT detection catches change points.

Methodology:
    Z-Score measure how many standard deviations a data point is fromt he mean. However it assumes Gaussian noise.
In our case, we are measuring how far the residual is from the mean.
z = (x - mean) / std

    IQR will flag values outside a given range: [Q1 - 1.5*IQR, Q3 + 1.5*IQR] and does not assume Gaussian noise.
However, it is less sensitive to outliers than Z-score (massive outliers will affect standard deviation more than a quartile).

    Source: https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/4_algos_and_penalties.html
    PELT (Pruned Exact Linear Time): In the past, to segment data according to change points, binary segmentation and optimal 
partitioning were used. However, BS is prone to error and OP is computationally expensive. PELT solves these issues by
"reduc[ing] the [number] of [changepoint] checks to be performed at each iteration," also called pruning. It introduces a penalty
to discourage the algorithm from continuosly adding changepoints (overfitting).
PELT
INPUT: Time series, penalty 
OUTPUT: Optimal changepoint vector 


'''


#Step 3: Implement PELT Change point model first so I can segment data for IQR and Z-score flagging

signal = residuals['p349_north']  

algo = rpt.Pelt(model="rbf").fit(signal)
breaks = algo.predict(pen = 10)


'''
I wrote:
print("Change points detected at indices:", breaks)
print(len(residuals['p349_north']))
and got:
7524
Change points detected at indices: [515, 895, 1565, 2695, 3485, 3770, 4415, 5235, 5600, 6125, 6375, 6855, 7070, 7524]
These are indices in the residual time series, in other words, these are EPOCHS at which a change point occurs.
So day 515, day 895, day 1565, etc. NOT frequencies.
Note, the last value is just the end of the time series, not a change point.
'''

changepoints = {}


for key, value in residuals.items():
    signal = value
    algo = rpt.Pelt(model="rbf").fit(signal)
    breaks = algo.predict(pen=10)
    
    station = key.split("_")[0]
    dates = station_dates[station]
    changepoints[key] = [dates[i] for i in breaks[:-1]]  # Exclude the last index which is the end of the series
    print(f"Change points detected for {key} at dates:", changepoints[key])

'''
My initial penalty is way too permissive. 
Source: https://academic.oup.com/gji/article/204/1/480/635055?login=false
With >700 days of data, a given station might have 1.8 changepoints on average. 
49% caused by documented equipment changes, 31% by earthquakes, and 20% due to unknown causes."
NOTE: Next, I will experiment with penalty values and create charts and persistent storage of results.
'''

#Step 4: Implement IQR and Modified Z-score Flagging  