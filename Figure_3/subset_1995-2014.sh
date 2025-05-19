#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=subset_mean_ts_so
#SBATCH -o job_output_files/subset_so_1993_2023.job
#SBATCH --time=120
#SBATCH --partition=batch

# Setup Env
module load nco

# Subset data
ncks -d time,2,21 ${1}/nep_so_1993-2023_mean_ts.nc ${1}/nep_so_1995-2014_mean_ts.nc
