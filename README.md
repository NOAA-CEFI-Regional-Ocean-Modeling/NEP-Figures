# NEP Diagnostics
This repo contains Jupyter Notebooks used to generate Figures 1-25 from the paper "A regional physical-biogeochemical ocean model for marine resource applications in the Northeast Pacific (MOM6-COBALT-NEP10k v1.0)", submitted to GMD.

## Python Notebooks

Most notebooks are written in python, and can be run using the MED maintained python environement. If you run the noteboooks PPAN, you can access this environment by running:
```
module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
```
Otherwise, feel free to install this environment on your system using this [yaml file](https://github.com/NOAA-CEFI-Regional-Ocean-Modeling/MEDpy/blob/main/med_py311.yml).

## R Notebook
The notebook `Figure_19_20/Calculate_Coldpool_Areas.ipynb` is written in R, and makes use of the R libraries `coldpool` and `akgfmaps`. MED currently provides a minimal R environment installed via conda that should be capable of running these notebooks. You can activate this environment on PPAN via the following commands:
```
module load miniforge
conda activate /work/role.medgrp/conda/r_env/
```
If you would like to install this environment for yourself, we also provide a [yaml file](https://github.com/NOAA-CEFI-Regional-Ocean-Modeling/MEDpy/blob/main/r_env.yml) with the necessary `R` packages you would need to install from conda before installing `coldpool` and `akfgmaps` using the [instructions](https://github.com/afsc-gap-products/coldpool?tab=readme-ov-file) provided in the coldpool repo.
