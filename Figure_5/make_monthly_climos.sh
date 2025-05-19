#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=monthly_climo_ssh
#SBATCH -o job_output_files/monthly_climo_ssh_1993_2019_%j.job
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

# Define variables
archive_dir="/archive/Utheri.Wagura/${prefix}/${experiment}/gfdl.ncrc6-intel23-prod/pp/ocean_monthly/ts/monthly/65yr/"

# Move files from archive to tmp, then concat and move them output
echo "Dmgetting"
dmget ${archive_dir}/ocean_monthly.196001-202412.ssh.nc
echo "GCPing"
gcp --sync -cd ${archive_dir}/ocean_monthly.196001-202412.ssh.nc ${TMPDIR}/ssh/
echo "subsetting from 1993 to 2019"
ncks -d time,396,719 ${TMPDIR}/ssh/ocean_monthly.196001-202412.ssh.nc ${TMPDIR}/ssh/ssh_1993-2019.nc
echo "taking annual mean, writing to work"
cdo ymonmean ${TMPDIR}/ssh/ssh_1993-2019.nc ${output_dir}/nep_ssh_1993-2019_clim.nc
