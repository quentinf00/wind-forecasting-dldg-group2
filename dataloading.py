import time
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.distributed import Client, progress
import xarray as xr
import os
import pandas as pd
import torch.utils.data as td
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.distributed import Client, progress
from pathlib import Path
import time

root_dir = Path(os.environ['HOME']) / 'local'

data_dir = root_dir / 'DL_GS_Project_STForecasting'
ground_station_dir = data_dir / 'ground_stations'
model_dir = data_dir / 'models'
mask_dir = data_dir / 'mask'

client = Client(n_workers=18, threads_per_worker=2, memory_limit='2GB')

df = dd.read_csv(ground_station_dir /'*.csv')
ds = xr.open_mfdataset(sorted(model_dir.glob('*')), engine='netcdf4')

cfg = {
        'items':[
            {'agg': 7 * 24 * 60 // 6 , 'size': 1},
            {'agg': None, 'size': 60},
            {'agg': '1H', 'size': 24},
        ]
}

date = pd.to_datetime('2016-01-15')
max_horizon = 1 * pd.to_timedelta('7D')
df['dt'] = dd.to_datetime(df.date)
item_ar = ds.pipe(lambda _ds: _ds.sel(valid_time=slice(date - max_horizon, date))).load()

item_gs = (
        df
        .assign(dt=lambda _df: dd.to_datetime(_df.date))
        # .set_index('dt')
        .loc[lambda _df:  (_df.dt<=date) ]
        .loc[lambda _df:  (_df.dt>=(date-max_horizon)) ]
).compute()

gs_ds = (
    item_gs.assign(
        g_lat=lambda _df: item_ar.latitude.data[::-1][np.searchsorted(item_ar.latitude.data[::-1], _df.lat, 'right')-1],
        g_lon=lambda _df: item_ar.longitude.data[np.searchsorted(item_ar.longitude.data, _df.lon, 'right')],
        date=lambda _df: pd.to_datetime(_df.date)
    ).assign(
        u=lambda _df: _df.ff * np.cos(_df.dd / 360 * 2 * np.pi),
        v=lambda _df: _df.ff * np.sin(_df.dd / 360 * 2 * np.pi),
    ).groupby(['g_lat', 'g_lon', 'date']).mean()[['u', 'v']]
    .pipe(xr.Dataset.from_dataframe)
)
item_ds = xr.merge([
    item_ar[['u10', 'v10']].interp(valid_time=gs_ds.date.values),
    gs_ds.rename({'g_lat': 'latitude', 'g_lon': 'longitude', 'date': 'valid_time'})
])
item1, item2, item3 = item_ds.assign(
        du=lambda _df: (_df.u - _df.u10).fillna(0.),
        dv=lambda _df: (_df.v - _df.v10).fillna(0.),
    ).pipe( lambda _ds: (
    _ds.isel(valid_time=slice(-60, None)),
    _ds.coarsen(valid_time=10, latitude=2, longitude=2, boundary='trim').mean().isel(valid_time=slice(-24, None)),
    _ds.coarsen(valid_time=7 * 24 * 60 // 6, latitude=4, longitude=4, boundary='trim').mean().isel(valid_time=slice(-1, None)),
    )
)

class WindForecastDataset(td.Dataset):
    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self):
        ...

if __name__ == '__main__':
    gs_ds = GroundStationDataset()

    gs_ds[0]
