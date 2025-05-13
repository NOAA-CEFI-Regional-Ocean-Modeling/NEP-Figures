#!/bin/bash
#SBATCH --output=job_output_files/ACTUAL_INDEX_Calc_Coldpool_2021_2024_%j.job
#SBATCH --job-name=ACTUAL_INDEX_Calc_Coldpool_2021_2024
#SBATCH --time=180
#SBATCH --partition=analysis
#SBATCH -n 1

source $MODULESHOME/init/bash
module load miniforge
#conda activate /work/role.medgrp/conda/r_env
conda activate /net3/e1n/miniconda3/envs/r_env # note that this env does not work on the batch partition

Rscript Calculate_Coldpool_Areas.r
