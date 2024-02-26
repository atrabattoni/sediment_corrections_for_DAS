import pandas as pd
import xarray as xr
from xloc.ttlut import get_grid, get_model, get_ttlut

resolution = 3_000.0

model = pd.read_csv("data/marot2014.csv", index_col="depth", dtype="float").to_xarray()
receivers = xr.open_dataset("data/fiber.nc")
stations = xr.open_dataarray("data/stations.nc")

stations["distance"] = stations["distance"]

receivers = receivers.interp_like(stations)

grid = get_grid(
    lon_lim=(-72.86, -70.66),
    lon_res=resolution / 100_000.0,
    lat_lim=(-33.80, -30.96),
    lat_res=resolution / 100_000.0,
    dep_lim=(-5_000.0, 100_000.0),
    dep_res=resolution,
)

model = get_model(
    model,
    dep_lim=(-5_000, 200_000.0),
    dep_res=1_000.0,
    dst_max=2_000_000.0,
    dst_res=1_000.0,
    tabular=True,
    spheric=True,
)

ttlut = (
    get_ttlut(receivers, grid, model, spheric=True)
    .to_array("phase")
    .assign_coords(station=("distance", stations.values))
    .swap_dims({"distance": "station"})
    .transpose("station", "phase", "longitude", "latitude", "depth")
)
ttlut.to_netcdf("/ssd/trabatto/sediment_corrections/ttlut.nc")
