import xarray
import os

# Set variables
tmp = os.getenv("TMPDIR")
experiment = "glorys_obc"
var = "so"
start = "1993"
end = "2023"

# All the data
print(f"Opening dataset from tmp: {tmp}/nep_{var}_{start}-{end}_ts.nc")
ds = xarray.open_dataset(f"{tmp}/nep_{var}_{start}-{end}_ts.nc")

print(f"Writing annual means from {start}-{end}")
# Write annual means for {start}-{end} period
ds.groupby("time.year").mean().to_netcdf(f"{experiment}/nep_{var}_{start}-{end}_mean_ts.nc")

print(f"Writing annual means from 1995-2014")
# Write annual means for 1995-2014 period
ds.sel( time=slice("1995-01-01","2014-12-31") ).groupby("time.year").mean().to_netcdf(f"{experiment}/nep_{var}_1995-2014_mean_ts.nc")
