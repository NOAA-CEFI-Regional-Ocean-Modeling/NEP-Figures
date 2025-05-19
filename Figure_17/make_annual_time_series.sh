#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=monthly_ts_tob
#SBATCH -o job_output_files/monthly_tob_and_sst_ts_1993_2019_%j.job
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

# Define variables
echo "Working on sst"
archive_dir="/archive/Utheri.Wagura/${prefix}/${experiment}/gfdl.ncrc6-intel23-prod/pp/ocean_monthly/ts/monthly/65yr/"

# Move files from archive to tmp, then concat and move them output
echo -e "\tDmgetting"
dmget ${archive_dir}/ocean_monthly.196001-202412.tos.nc
echo -e "\tGCPing"
gcp --sync -cd ${archive_dir}/ocean_monthly.196001-202412.tos.nc ${TMPDIR}/tos/
echo -e "\tsubsetting"
ncks -d time,396,719 ${TMPDIR}/tos/ocean_monthly.196001-202412.tos.nc ${output_dir}/nep_sst_1993-2019_monthly_ts.nc

# Define variables
echo "Working on tob"

# Move files from archive to tmp, then concat and move them output
echo -e "\tDmgetting"
dmget ${archive_dir}/ocean_monthly.196001-202412.tob.nc
echo -e "\tGCPing"
gcp --sync -cd ${archive_dir}/ocean_monthly.196001-202412.tob.nc ${TMPDIR}/tob/
echo -e "\tsubsetting form 1993 to 2019"
ncks -d time,396,719 ${TMPDIR}/tob/ocean_monthly.196001-202412.tob.nc ${output_dir}/nep_tob_1993-2019_monthly_ts.nc
