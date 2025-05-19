#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=ncks_ts_tob
#SBATCH -o job_output_files/ncks_tob_ts_1993_2024_%j.job
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
tob_dir="/archive/Utheri.Wagura/${prefix}/${experiment}/gfdl.ncrc6-intel23-prod/pp/ocean_daily/ts/daily/64yr/"

# Move files from archive to tmp, then concat and move them output
echo "Dmgetting"
dmget ${tob_dir}/ocean_daily.19600101-20231231.tob.nc
echo "GCPing"
gcp --sync -cd ${tob_dir}/ocean_daily.19600101-20231231.tob.nc ${TMPDIR}/nep_tob_1960-2024_daily_ts.nc
echo "subsetting from 1993 to present"
ncks -d time,11490,-1 ${TMPDIR}/nep_tob_1960-2024_daily_ts.nc ${TMPDIR}/nep_tob_1993-2024_daily_ts.nc
ncks -O -d yh,560,-107 -d xh,66,215 ${TMPDIR}/nep_tob_1993-2024_daily_ts.nc ${output_dir}/nep_tob_bering_1993-2024_daily_ts.nc
