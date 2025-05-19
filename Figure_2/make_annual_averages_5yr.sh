#!/bin/bash

# Setup Env
module load nco

# Define variables
pp_dir=""
domain=
output_dir=""

# Move files from archive to tmp, then concat and move them output
for decade in 199 200 210 ; do
    echo "Working on the ${decade}0s"
    cp ${pp_dir}/${domain}/
    #ncrcat /archive/Utheri.Wagura/fre/NEP/2025_03/NEP10_PHYSICS_decr7_e${ens}/gfdl.ncrc6-intel23-prod/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.197*.thetao.nc ./nep_theta_1970s_ens_${ens}.nc
    #ncra -O -F -d time,01,,12
done

# Concatenate files

# Make climatology

# Output: nep_thetao_003_1993-2019_clim.nc
