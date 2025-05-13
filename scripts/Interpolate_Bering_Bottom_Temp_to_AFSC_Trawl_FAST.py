#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the python packages you will need 
import pandas 
import numpy as np
import netCDF4 as nc
import xarray as xr
import xesmf
import time
import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--experiment', type=str, help='experiment name', required=True)
args = parser.parse_args()
experiment = args.experiment


# In[2]:


# NEP Bering Grid 
bering_nep_grid_file = '/work/role.medgrp/NEP/plotting/Figure_19_20/nep_bering_grid.nc'
grd_fid = nc.Dataset(bering_nep_grid_file)

# extracting lat/lon from geolat/geolon
nep_lat = grd_fid.variables['geolat'][:]
nep_lon = grd_fid.variables['geolon'][:]


# #### Interpolation for Eastern and Northern Bering Sea for Kearney spatial plotting

# In[3]:


# daily NEP bering sea bottom temp
#nep_fil = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/nep_tob_bering_1993-2019_daily_ts.nc'
nep_fil = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/nep_tob_bering_1993-2024_daily_ts.nc'
nep_ds = xr.open_dataset(nep_fil)
nep_ds = nep_ds.assign_coords(lon = (('yh', 'xh'), nep_lon))
nep_ds = nep_ds.assign_coords(lat = (('yh', 'xh'), nep_lat))

print(nep_ds)

## daily bottom temp
trawl_fil = '/work/role.medgrp/NEP/TOOLS/coldpool/data/ebs_nbs_temperature_full_area.csv'
df = pandas.read_csv(trawl_fil)

# drop rows that go back for forward in time further than the temporal extent of the NEP hindcast
df = df[df.year >= 1993].reset_index(drop=True)
#df = df[df.year <= 2019].reset_index(drop=True)

# only measurements with haul type 3
df = df[df.haul_type == 3].reset_index(drop=True)

# generate copy of trawl dataframe with added empty column for interpolated NEP temp
df2 = df.assign(nep_tob=np.nan*np.zeros(len(df)))


# In[4]:


# convert time to ALASKA time
times = np.array(df['start_time'].to_list(), dtype="datetime64")
times_alaska = times + np.timedelta64(8,"h")

trawl_locs = xr.Dataset()
trawl_locs['lon'] = xr.DataArray(data=df['longitude'].to_list(), dims=("trawl_sample") )
trawl_locs['lat'] = xr.DataArray(data=df['latitude'].to_list(), dims=("trawl_sample") )
print(trawl_locs)

 # Define regridder
nep_regridder = xesmf.Regridder(nep_ds, trawl_locs , 'bilinear', locstream_out=True)

# Interpolate times
nep_ds_interp = nep_ds.interp(time=times_alaska)

# Regrid
nep_regrided = nep_regridder(nep_ds_interp)

print(nep_regrided)

for i in range( len(df2)):
    df2['nep_tob'][i] = nep_regrided.tob.isel(time=i,trawl_sample=i)
 
new_csv = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/ebs_nbs_temperature_full_area_nep.csv'
df2.to_csv(new_csv)
df2['nep_tob']


# In[ ]:





# #### Interpolation for Index Bering Sea for Rohan CPA index processing and analysis

# In[5]:


## daily bottom temp
trawl_fil = '/work/role.medgrp/NEP/TOOLS/coldpool/data/index_hauls_temperature_data.csv'
df = pandas.read_csv(trawl_fil)
# drop rows that go back for forward in time further than the temporal extent of the NEP hindcast
df = df[df.year >= 1993].reset_index(drop=True)
#df = df[df.year <= 2019].reset_index(drop=True)

# append fake 2020 time values for nep regridding
df_2020 = df[df.year==2019].reset_index(drop=True)
df_2020.year = 2020
df_2020.gear_temperature = np.nan
df_2020.surface_temperature  = np.nan
df_2020.cruise = np.nan
for row in df_2020.itertuples():
    index=row[0]
    time_str = df_2020.start_time[index].split('2019')[1] 
    df_2020.start_time[index] = ('2020' + time_str)

df = pandas.concat([df, df_2020]).reset_index(drop=True)

nep_fil = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/nep_tob_bering_1993-2024_daily_ts.nc'
#nep_fil = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/nep_tob_bering_1993-2019_daily_ts.nc'
nep_ds = xr.open_dataset(nep_fil)
nep_ds = nep_ds.assign_coords(lon = (('yh', 'xh'), nep_lon))
nep_ds = nep_ds.assign_coords(lat = (('yh', 'xh'), nep_lat))

# generate copy of trawl dataframe with added empty column for interpolated NEP temp
df2 = df.assign(nep_tob=np.nan*np.zeros(len(df)))

trawl_locs = xr.Dataset()
# get lat lon values for regridding
trawl_locs['lon'] = xr.DataArray(data=df['longitude'].to_numpy(), dims=('trawl_sample'))
trawl_locs['lat'] = xr.DataArray(data=df['latitude'].to_numpy(), dims=('trawl_sample'))

# Get time values for time interpolation
print("Creating array of Alaska times:")
times = np.array(df['start_time'].to_list(), dtype="datetime64")
times_alaska = times + np.timedelta64(8,"h")

# Create regridder to interp latlon to to trawl locations using xesmf
print("Creating regridder to trawl locations:")
nep_regridder = xesmf.Regridder(nep_ds, trawl_locs, 'bilinear', locstream_out=True)

# Interpolate time to Alaska Daylight Time: UTC-8
print("Performing interpolation: ")
nep_interpreted = nep_ds.interp(time=times_alaska)
print("Interpolated times: ")

# regrid NEP and GLORYS bottom temperature value
print("Regridding to trawl locations:")
nep_regridded = nep_regridder(nep_interpreted)
print("Regrided dataset:")

# Write interpreted values to dataframe
print("Writing interpolated values to dataframe")
for i in range( len(df2) ):
    df2['nep_tob'][i] = nep_regridded.tob.isel(time=i,trawl_sample=i)

print("Saving dataframe")
new_csv = f'/work/Utheri.Wagura/NEP/plotting/Figure_19_20/{experiment}/index_hauls_temperature_data_nep.csv'
df2.to_csv(new_csv)
df2['nep_tob']

