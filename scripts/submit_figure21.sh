#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_21
#SBATCH -o job_output_files/make_figure21_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate medpy311

python Figure21_Bering_Sea_sisconc_extent.py -e ${1}
