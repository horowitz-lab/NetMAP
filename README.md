The related publication is: https://www.nature.com/articles/s41598-023-50089-1. Please cite it if you use this code!



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
