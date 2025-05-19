#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=monthly_mean_ts_thetao
#SBATCH -o job_output_files/monthly_mean_ts_thetao_1993_2024_%j.job
#SBATCH --time=150
#SBATCH --partition=batch

# Setup Env
module load gcp
module load nco
module load cdo

# Define variables
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

archive_dir="/archive/Utheri.Wagura/${prefix}/${experiment}/gfdl.ncrc6-intel23-prod/pp/ocean_monthly_z/ts/monthly/65yr/"

# Move files from archive to tmp, then concat and move them output
echo "Dmgetting"
dmget ${archive_dir}/ocean_monthly_z.196001-202412.thetao.nc
echo "GCPing"
gcp --sync -cd ${archive_dir}/ocean_monthly_z.196001-202412.thetao.nc ${TMPDIR}/thetao/
echo "subsetting"
ncks -d time,396,-1 ${TMPDIR}/thetao/ocean_monthly_z.196001-202412.thetao.nc ${TMPDIR}/thetao/thetao_1993-2024.nc
echo "taking annual mean, writing to work"
cdo yearmonmean ${TMPDIR}/thetao/thetao_1993-2024.nc ${output_dir}/nep_thetao_1993-2024_mean_ts.nc
echo "Extracting data up to 2023"
ncks -d time,,-2 ${output_dir}/nep_thetao_1993-2024_mean_ts.nc ${output_dir}/nep_thetao_1993-2023_mean_ts.nc
