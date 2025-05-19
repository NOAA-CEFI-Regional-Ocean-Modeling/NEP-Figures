import xarray
import os

# Set variables
tmp = os.getenv("TMPDIR")
experiment = "glorys_obc"
var = "ssh"
start = "1993"
end = "2023"
end_sel = "2019"

# All the data
print(f"Opening dataset from tmp: {tmp}/nep_{var}_{start}-{end}_ts.nc")
ds = xarray.open_dataset(f"{tmp}/nep_{var}_{start}-{end}_ts.nc").sel( time = slice("1993-01-01",f"2019-12-31") )

print(f"Writing monthly climo from {start}-{end}")
# Write annual means for {start}-{end} period
ds.groupby("time.month").mean().squeeze().to_netcdf(f"{experiment}/nep_{var}_{start}-{end_sel}_clim.nc")
