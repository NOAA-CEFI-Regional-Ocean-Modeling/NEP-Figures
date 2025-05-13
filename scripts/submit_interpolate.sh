#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_interpolation
#SBATCH -o job_output_files/make_figure_interpolation_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate medpy311

python Interpolate_Bering_Bottom_Temp_to_AFSC_Trawl_FAST.py -e ${1}
