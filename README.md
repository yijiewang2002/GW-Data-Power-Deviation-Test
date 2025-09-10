# GW Data Power Deviation Test

This repository contains a collection of scripts developed to compute and analyze the **GW data power** \(d,d\), a key quantity in gravitational-wave (GW) data analysis. The statistic \(d,d\) is the **noise-weighted norm** of detector strain data and plays an important role in understanding detection thresholds, likelihood formulations, and statistical properties of GW events.

$$(d,d) =4 \text{Re} \int_0^\infty \frac{ |n(f)+s(f)|^2 }{S_n(f)} df$$

---

## Download

```bash
# 1) Clone WITH submodules (required for gw-detectors)
git clone --recurse-submodules https://github.com/yijiewang2002/GW-Data-Power-Deviation-Test.git

# 2) Enter the project
cd GW-Data-Power-Deviation-Test

# 3) Make the submodule importable (editable/dev mode)
pip install -e gw-detectors/
```

## Dependencies
h5py

numpy

scipy

matplotlib

pycbc

## Script Description
data_norm.py: calculates the data power of a given event

extract_snr.py: extracts the observed and optimal SNR from a parameter estimation sample associated with an event

data_simulation.py: simulates GW events and plots the residual power of these simulated events along with the residual power of five real events
