#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_17
#SBATCH -o job_output_files/make_figure17_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate medpy311

python Figure17_NEP_GLORYS_temperature_anomaly_timeseries.py -e $1 -y $2
