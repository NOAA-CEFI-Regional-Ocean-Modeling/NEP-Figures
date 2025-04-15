#!/bin/bash
#SBATCH --output=job_output_files/calculate_coldpool_areas_%j.job
#SBATCH --job-name=Calculate_coldpool_areas
#SBATCH --time=300
#SBATCH --partition=batch
#SBATCH -n 1

source $MODULESHOME/init/bash
module load miniforge
conda activate /work/role.medgrp/conda/r_env

Rscript Calculate_Coldpool_Areas.r
