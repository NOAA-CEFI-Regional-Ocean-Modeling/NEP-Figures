#!/bin/bash
#SBATCH --output=job_output_files/no_bias_correction_coldpool_%j.job
#SBATCH --job-name=no_bias_correction_coldpool
#SBATCH --time=300
#SBATCH --partition=batch
#SBATCH -n 1

source $MODULESHOME/init/bash
module load miniforge
conda activate /work/role.medgrp/conda/r_env
#conda activate /net3/e1n/miniconda3/envs/r_env # note that this env does not work as a batch job

Rscript Calculate_Coldpool_Areas.r
