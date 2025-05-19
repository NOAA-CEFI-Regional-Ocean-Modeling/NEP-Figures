#!/bin/bash

for ens in {1..5}; do
    echo "Working on ${ens}"
    ncrcat /archive/Utheri.Wagura/fre/NEP/2025_03/NEP10_PHYSICS_decr7_e${ens}/gfdl.ncrc6-intel23-prod/pp/ocean_monthly/ts/monthly/5yr/ocean_monthly.*.MLD_003.nc ./nep_MLD_003_1970s_ens_${ens}.nc
    ls
    #ncra -O -F -d time,01,,12
done

# Output: nep_MLD_003_1993-2019_clim.nc
