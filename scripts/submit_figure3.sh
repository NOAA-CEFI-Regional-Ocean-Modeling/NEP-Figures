#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_3
#SBATCH -o job_output_files/make_figure3_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311

python Figure3_NEP_NCEI_GLORYS_salinity_comparison.py -e ${1}
