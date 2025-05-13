#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_5
#SBATCH -o job_output_files/make_figure5_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311

python Figure5_SSH_comparisons.py -e $1
