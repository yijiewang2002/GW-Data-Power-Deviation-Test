# GW Data Power

This repository contains a collection of scripts developed to compute and analyze the **data power statistic** \(d,d\), a key quantity in gravitational-wave (GW) data analysis. The statistic \(d,d\) is the **noise-weighted norm** of detector strain data and plays an important role in understanding detection thresholds, likelihood formulations, and statistical properties of GW events.

$$
(d,d) = 4\,\Re \int_{f_{l}}^{f_{u}} \frac{|\tilde d(f)|^2}{S_n(f)}\,df
$$

---

## Download

```bash
# 1) Clone WITH submodules (required for gw-detectors)
git clone --recurse-submodules https://github.com/yijiewang2002/GW-Data-Power-Deviation-Test.git

# 2) Enter the project
cd GW-Data-Power-Deviation-Test

# 3) Make the submodule importable (editable/dev mode)
pip install -e gw-detectors/
