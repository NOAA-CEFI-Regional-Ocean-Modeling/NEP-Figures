#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=monthly_ts_tob
#SBATCH -o job_output_files/monthly_siconc_ts_1993_2019_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

# Setup Env
module load gcp
module load nco

while getopts "e:p:t:d:" opt; do
  case $opt in
      e) experiment="$OPTARG" ;;
      p) prefix="$OPTARG" ;;
      t) time_span="$OPTARG" ;;
      d) output_dir="$OPTARG" ;;
      \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
      :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

set -eu

# Define variables
archive_dir="/archive/Utheri.Wagura/${prefix}/${experiment}/gfdl.ncrc6-intel23-prod/pp/ice_monthly/ts/monthly/65yr/"

# Move files from archive to tmp, then concat and move them output
echo "Dmgetting"
dmget ${archive_dir}/ice_monthly.196001-202412.siconc.nc
echo "GCPing"
gcp --sync -cd ${archive_dir}/ice_monthly.196001-202412.siconc.nc ${TMPDIR}/siconc/
echo "subsetting form 1993 to 2019"
ncks -d time,396,719 ${TMPDIR}/siconc/ice_monthly.196001-202412.siconc.nc ${output_dir}/nep_siconc_1993-2019_ts.nc
