#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=make_figure_19
#SBATCH -o job_output_files/make_figure19_and_20_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

source $MODULESHOME/init/bash
module load miniforge
conda activate medpy311

python Figure19-20_Bering_Cold_Pool_Analyses.py -e $1
