# GNSS Position Time Series Analysis
**June 2026**

---

## Overview

This project quantifies how much least squares underestimates uncertainty when noise is colored rather than white/Gaussian/independent. It performs statistical decomposition and noise characterization of ground motion time series from four continuously operating GNSS (Global Navigation Satellite System) stations in the Pacific Northwest and Northern California. Data is sourced from the EarthScope GAGE facility, processed by Central Washington University (CWU) in the NAM14 North America-fixed reference frame.

Motivated by the geodetic signal processing research of Dr. Amanda Thomas's Observational Seismology Research Group at UC Davis.

---

## Stations

| Station | Location | Record Start |
|--------|----------|--------------|
| P349 | Shasta Lake, CA | 2005 |
| P380 | Klamath Falls, OR | ~2007 |
| P434 | Stevens Pass, WA | 2008 |
| P441 | Kendall, WA | ~2007 |

Each station records daily displacement in three directions — North, East, and Vertical — in millimeters relative to a reference position. The NAM14 reference frame removes background North American plate motion, isolating residual signals of interest.

---

## What This Project Does

### Step 1 — Signal Decomposition via Least Squares
Each displacement time series is modeled as a sum of physical components:

```
position(t) = a + b·t + c·sin(2πt) + d·cos(2πt) + e·sin(4πt) + f·cos(4πt) + residual
```

Where:
- `a` — initial offset
- `b·t` — linear tectonic velocity (mm/year)
- `c, d` — annual seasonal signal (snow loading, hydrology)
- `e, f` — semi-annual seasonal signal
- `t` — decimal years since first observation

A design matrix X is constructed explicitly using NumPy and solved with `np.linalg.lstsq` — raw linear algebra, no ML libraries. This is run independently for North, East, and Vertical at each of the four stations, yielding 12 sets of coefficients total.

### Step 2 — Noise Characterization
GNSS residuals are not white noise. This step characterizes the colored noise structure by computing the Power Spectral Density (PSD) of residuals and analyzing the log-log slope:

- **White noise** — flat spectrum, slope ≈ 0
- **Flicker noise** — slope ≈ -1, correlated across time
- **Random walk** — slope ≈ -2, strongest long-period correlations

This is the same noise characterization problem worked on operationally at NASA Goddard and JPL.

### Step 3 — Outlier and Change-Point Detection
Residuals exceeding 3σ are flagged as outliers. The PELT algorithm (`ruptures` library) is applied to automatically detect position discontinuities — jumps in the time series caused by equipment changes or seismic events. Detected change-points are compared against documented offsets in `metadata.json`.

### Step 4 — Velocity Uncertainty Quantification
Demonstrates that assuming white noise significantly underestimates velocity uncertainty. Quantifies the difference numerically under a realistic flicker + random walk noise model — the core statistical contribution of the project.

### Step 5 — Visualization and Report
Clean plots of raw time series, decomposed components, residuals, PSD, and change-points for all stations. Accompanied by a 4-page technical report covering methodology, assumptions, results, and limitations.

---

## Data Source

Downloaded from the EarthScope GAGE facility (`gage-data.earthscope.org`). Files follow the naming convention `P***.cwu.nam14.csv`.

---

## File Structure

```
WIP
```

---

## Dependencies

```
pandas
numpy
scipy
ruptures
matplotlib
```

Chelsea Momoh
Statistics, UC Davis
