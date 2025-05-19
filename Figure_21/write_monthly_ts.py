import xarray
import os

# Set variables
tmp = os.getenv("TMPDIR")
experiment = "glorys_obc"
var = "siconc"
var1_name = "siconc"
start = "1993"
end = "2023"
end_sel = "2019"

# First var
print(f"Opening dataset from tmp, subsetting, and writing : {tmp}/nep_{var}_{start}-{end}_ts.nc")
ds = xarray.open_dataset(f"{tmp}/nep_{var}_{start}-{end}_ts.nc").sel( time = slice("1993-01-01",f"2019-12-31") )
ds.to_netcdf(f"{experiment}/nep_{var1_name}_{start}-{end_sel}_ts.nc")
