The related publication is: https://www.nature.com/articles/s41598-023-50089-1. Please cite it if you use this code!

# NetMAP

[The following documentation was written with AI.]

NetMAP is an algebraic framework designed for the rapid characterization of mechanical resonator networks (monomers, dimers, and higher-order systems). It allows researchers to extract physical parameters—such as mass ($m$), damping ($b$), and stiffness ($k$)—directly from experimental frequency response data (spectra) using linear algebra rather than iterative non-linear fitting.

## Overview
Characterizing nanomechanical or micromechanical resonators often requires fitting complex models to frequency sweeps. NetMAP simplifies this by transforming the differential equations of motion into a linear system $Z\vec{p} = 0$, where $Z$ is a "Z-matrix" constructed from measured complex amplitudes and $\vec{p}$ is a vector of the unknown physical parameters.

By applying **Singular Value Decomposition (SVD)** to the Z-matrix, the system's physical constants can be recovered from the nullspace, even in the presence of experimental noise.

## Key Features
* **Algebraic Parameter Recovery:** No need for "guesses" or iterative fitting; parameters are recovered through direct matrix decomposition.
* **Support for Dimer/Coupled Systems:** Specifically designed to handle coupled resonators (e.g., graphene drums or cantilever arrays) where traditional peak-fitting fails due to overlapping modes.
* **Noise Robustness:** Evaluates recovery accuracy across 1D, 2D, and 3D nullspace assumptions to find the most stable physical solution.
* **Synthetic Validation:** Includes a `simulated_experiment` suite to validate the framework against "ground truth" models with controlled noise injection.

## Core Components
* `Zmat()`: The primary engine for constructing the Z-matrix from Pandas DataFrames containing frequency, amplitude, and phase data.
* **Monomer & Dimer Logic:** Dedicated matrix construction for single oscillators or two-resonator coupled systems.
* **Normalization Tools:** Functions to scale relative SVD results into absolute physical units (kg, N/m, N s/m) based on a known reference (like a known driving force or mass).

## Scientific Context
This codebase is used for validating algebraic approaches to characterizing resonator networks, as described in:
> *Horowitz et al., "Validating an algebraic approach to characterizing resonator networks," Scientific Reports 14, 1325 (2024).*

## Getting Started
### Dependencies
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`

### Basic Workflow
1. Load your frequency sweep data into a DataFrame.
2. Pass the data to `Zmat()` to generate the Z-matrix.
3. Perform SVD (`np.linalg.svd`) to identify the parameter ratios.
4. Normalize the result using `normalize_parameters_1d_by_force` (or similar) to extract final $m, b, k$ values.

## Function Documentation: `simulated_experiment()`

[The following documentation was written with AI.]

The `simulated_experiment()` function is a comprehensive validation tool for the NetMAP framework. It performs "end-to-end" testing by generating synthetic resonance data (with controlled noise), applying the NetMAP Singular Value Decomposition (SVD) recovery process, and statistically comparing the recovered physical parameters against the original "ground truth" values. This was the essential tool for the validation in the paper https://www.nature.com/articles/s41598-023-50089-1.

---

### Description
This function simulates a physical measurement of a resonator (monomer or dimer), introduces noise, and attempts to solve the inverse problem—extracting mass ($m$), damping ($b$), and stiffness ($k$) from the resulting spectra. It evaluates the success of the recovery across different assumptions of the SVD nullspace (1D, 2D, and 3D).

---

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `measurementfreqs` | list/array | The specific frequencies ($\omega$) where "measurements" are taken. |
| `vals_set` | list | The "ground truth" parameters: $[m_1, m_2, b_1, b_2, k_1, k_2, k_{12}, F_{set}]$. |
| `noiselevel` | float | Magnitude of Gaussian noise added to the complex amplitude. |
| `MONOMER` | bool | `True` for a single resonator; `False` for a coupled dimer. |
| `forceboth` | bool | If `True`, applies the driving force to both masses in a dimer. |
| `repeats` | int | Number of simulation iterations (useful for Monte Carlo error analysis). |
| `verbose` | bool | If `True`, prints detailed recovery statistics and SVD descriptions. |
| `demo` | bool | If `True`, formats plots for presentation (removes axis ticks). |

---

### Logic Flow

1.  **Spectrum Generation:** Calculates a noiseless baseline using `calculate_spectra()`. 
2.  **Noise Injection:** Applies Gaussian noise to the complex amplitude response based on the `noiselevel`.
3.  **Z-Matrix Construction:** Calls `Zmat()` (see below) to build the linear system of equations based on the "noisy" measurement points.
4.  **SVD Recovery:** * Performs Singular Value Decomposition on the Z-matrix.
    * Identifies the singular vector corresponding to the smallest singular value (the suspected solution).
5.  **Normalization & Scaling:** Since SVD returns relative ratios, the function normalizes the vector using three different strategies:
    * **1D:** Normalizes based on the known force ($F_{set}$).
    * **2D:** Normalizes assuming two known parameters (e.g., $m_1$ and $F_{set}$).
    * **3D:** Normalizes assuming three known parameters.
6.  **Error Analysis:**
    * Calculates the **Systematic Error** (`syserr`) for every individual parameter.
    * Calculates $R^2$ values to determine how well the recovered model reconstructs the observed data.
7.  **Visualization:** If `verbose` is enabled, it calls `plot_SVD_results()` to overlay the recovered model onto the noisy data points.

---

### Return Values
The function primarily populates the `results` and `theseresults_cols` lists. These results include:
* **Recovered Parameters:** Values for $M, B, K,$ and $F$ for all normalization dimensions (1D, 2D, 3D).
* **Error Metrics:** Mean, RMS, and Max systematic errors for the recovery.
* **Spectral Data:** Signal-to-Noise Ratios (SNR) at each measurement frequency.

---

### Supporting Helper Functions
* **`describeresonator()`**: Utility to print a summary of the physical system's properties (Quality factor $Q$, peak width, etc.).
* **`measurementdfcalc()`**: Organizes raw frequency/amplitude/phase data into a formatted DataFrame for matrix processing.
* **`compile_rsqrd()`**: Calculates the goodness-of-fit for the recovered model in both polar and Cartesian coordinates.

---

### Usage Note
This function is essential for determining the **robustness** of NetMAP. By sweeping the `noiselevel` or varying the `measurementfreqs` (e.g., placing them on or off resonance), researchers can determine the optimal experimental conditions required to recover physical constants with high accuracy. See our publication in Scientific Reports.


## NetMAP Z-Matrix Documentation

[The following documentation is written with AI.]

In the NetMAP framework, the **Z-Matrix** is the linear algebraic representation of the system's equations of motion. These functions construct the matrix used for **Singular Value Decomposition (SVD)** or linear least-squares fitting to extract physical parameters (mass, damping, stiffness) from experimental frequency response data.

---

### `Zmat()`
**The primary entry point.** This wrapper function directs the data to either the Monomer or Dimer (2-resonator) matrix generator based on the system configuration.

| Parameter | Description |
| :--- | :--- |
| `measurementdf` | Pandas DataFrame containing experimental sweeps. |
| `MONOMER` | Boolean. If `True`, uses the single-oscillator model. |
| `forceboth` | Boolean. Relevant for dimers; indicates if both masses are driven. |
| `frequencycolumn` | Column name for the driving frequency ($\omega$). |
| `complexamplitudeX` | Column names for the complex response ($\tilde{Z} = A e^{i\phi}$). |

---

### `Zmatrix2resonators()`
Generates the Z-matrix for a **Dimer** system. For every frequency point provided, it generates **four rows** in the matrix (Real/Imaginary components for both Resonator 1 and Resonator 2).

**Matrix Structure:**
The columns correspond to the following physical parameters:
$\vec{p} = [m_1, m_2, b_1, b_2, k_1, k_2, k_{12}, F_1]$

**Logic:**
* **Driving Force:** If `forceboth` is `False`, only $m_1$ is driven ($F_1$ affects $R_1$). If `True`, a coupling factor `ff` is applied to the $R_2$ equations to account for the second drive.
* **Coupling:** The $k_{12}$ column accounts for the relative displacement $(ZZ_1 - ZZ_2)$.

**Mathematical Basis (per frequency $\omega$):**
For each resonator, the equation $[-\omega^2 m + i\omega b + k]\tilde{Z} = F$ is split into real and imaginary parts to populate the matrix.

---

### `ZmatrixMONOMER()`
Generates the Z-matrix for a **Monomer** system. For every frequency point, it generates **two rows** (Real and Imaginary).

**Matrix Structure:**
The columns correspond to:
$\vec{p} = [m_1, b_1, k_1, F]$

**Row Construction:**
1.  **Real Row:** $[- \omega^2 \cdot \text{Re}(Z), -\omega \cdot \text{Im}(Z), \text{Re}(Z), -1]$
2.  **Imaginary Row:** $[- \omega^2 \cdot \text{Im}(Z), \omega \cdot \text{Re}(Z), \text{Im}(Z), 0]$

---

### Summary of Parameter Mapping

| System Type | Columns (Parameters) | Rows per Frequency Point |
| :--- | :--- | :--- |
| **Monomer** | $m, b, k, F$ | 2 |
| **Dimer** | $m_1, m_2, b_1, b_2, k_1, k_2, k_{12}, F_1$ | 4 |

---

### Usage Example

```python
import numpy as np

# Assuming 'df' contains your resonance sweep data
z_matrix = Zmat(df, MONOMER=False, forceboth=True)

# Use SVD to solve for the physical parameters (p)
# Z * p = 0  -> The nullspace contains the parameter ratios
u, s, vh = np.linalg.svd(z_matrix)
parameters = vh[-1, :] # The last row of vh is the solution
```

## Function Documentation: `res_freq_numeric()`

[The following documentation was written with AI.]

The `res_freq_numeric()` function identifies resonance peak frequencies for physical systems (monomers or dimers) using numerical methods. It combines analytical weak-coupling approximations with iterative peak-finding on amplitude and phase curves.

---

### Description
This function calculates the resonant frequencies of a mechanical or electrical system. While optimized for **dimer** systems (two coupled oscillators), it also supports **monomer** systems. It uses an iterative refinement process to find peaks in amplitude response and specific crossings in phase response (e.g., $\pm\pi/2$).

---

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `vals_set` | dict/list | Parameters for the system (mass, damping, stiffness). |
| `MONOMER` | bool | If `True`, treats the system as a single oscillator. |
| `forceboth` | bool | Boolean flag passed to underlying curve functions to dictate forcing conditions. |
| `mode` | str | Method to find peaks: `'all'` (default), `'amplitude'`, or `'phase'`. |
| `minfreq` / `maxfreq` | float | The frequency range to search within. Defaults: `0.1` to `5`. |
| `numtoreturn` | int/None | Specific number of frequencies to return. If `None`, returns all found. |
| `iterations` | int | Number of times to refine the frequency mesh around suspected peaks. |
| `unique` / `veryunique` | bool | If `True`, filters out duplicate or near-identical frequencies. |
| `verbose` / `verboseplot` | bool | Enables console logging and Matplotlib visualization of peaks. |

---

### Logic Flow

1.  **Parameter Extraction:** Reads physical constants ($m, b, k, F$) using `read_params()`.
2.  **Initial Approximation:** Uses `res_freq_weak_coupling()` to find a starting point for the search.
3.  **Mesh Refinement:** * Generates a frequency list (`morefrequencies`).
    * Iteratively zooms in on frequencies where peaks are detected to improve precision.
4.  **Signal Calculation:** Generates Amplitude (`curve1`, `curve2`) and Phase (`theta1`, `theta2`) data across the mesh.
5.  **Peak Detection:** * **Amplitude:** Uses `find_peaks` to locate local maxima.
    * **Phase:** Uses `find_freq_from_angle` to locate resonant phase shifts ($\pm\pi/2$).
6.  **Selection & Filtering:** * If `veryunique` is active, it compares close peaks and keeps the one with the higher amplitude.
    * If `numtoreturn` is set (e.g., to 2), the function logic forces a selection of the most prominent peaks or pads the result with a frequency at a specific phase (-3π/4) if not enough peaks are found.

---

### Returns
* **freqlist** (`list` of `float`): A sorted list of identified resonant frequencies.
* **option_code** (`int`, optional): If `returnoptions=True`, returns an integer indicating which internal logic path (1–11) was used to finalize the list.

---

### Usage Example
```python
# Find the two primary resonance peaks for a dimer system
frequencies = res_freq_numeric(my_params, MONOMER=False, forceboth=True, numtoreturn=2)

print(f"Resonant Frequencies: {frequencies}")
