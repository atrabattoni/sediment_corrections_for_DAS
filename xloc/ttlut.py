import dask
import numpy as np
import pandas as pd
import xarray as xr
from fteikpy import Eikonal2D
from pyproj import Geod

r0 = 6371_000.0

# Build grid


def get_grid(lon_lim, lon_res, lat_lim, lat_res, dep_lim, dep_res):
    """
    Build 3D grid of possible source location.

    Parameters
    ----------
    lon_lim: tuple
        Minimum and maximum longitudes in degrees.
    lon_res: float
        Longitude resolution in degrees
    lat_lim: tuple
        Minimum and maximum latitudes in degrees.
    lat_res: float
        Latitude resolution in degrees
    dep_lim: tuple
        Minimum and maximum depths in meters.
    dep_res: float
        depth resolution in meters

    Returns
    -------
    dict[string, DataArray]
        The grid coordinates.
    """
    return {
        "longitude": _get_coord(lon_lim[0], lon_lim[1], lon_res, "longitude"),
        "latitude": _get_coord(lat_lim[0], lat_lim[1], lat_res, "latitude"),
        "depth": _get_coord(dep_lim[0], dep_lim[1], dep_res, "depth"),
    }


def _get_coord(xmin, xmax, res, name):
    n = round((xmax - xmin) / res) + 1
    x = np.linspace(xmin, xmax, n)
    return xr.DataArray(x, {name: x})


# Build model


def get_model(model, dep_lim, dep_res, dst_max, dst_res, tabular=False, spheric=False):
    """
    Build velocity model from 1D profile.

    Parameters
    ----------
    model: Dataset
        1D velocity model (m/s) as Dataset with variables "p" and/or "s" and dimension
        "depth" (meters).
    dep_lim: tuple[float]
        Minium and maximum source or receiver depth (meters). Negative values are
        above the sea level.
    dep_res: float
        Depth resolution (meters).
    dst_max: float
        Maximum source-receiver distance (meters).
    dst_res: float
        Distance resolution (meters):

    Returns
    -------
    Dataset
        Velocity model (m/s) as a Datase with variable "p" and/or "s" and dimensions
        ("depth", "distance").bj
    """
    dep = np.arange(dep_lim[0], dep_lim[1], dep_res)
    dst = np.arange(0.0, dst_max, dst_res)
    if tabular:
        method = "zero"
    else:
        method = "linear"
    model = model.interp(
        depth=dep, kwargs=dict(fill_value="extrapolate"), method=method
    )
    if spheric:
        model = model * r0 / (r0 - model["depth"])
        model["depth"] = r0 * (1 - np.exp(-model["depth"] / r0))
    model = model.expand_dims(distance=dst, axis=-1)
    return model


# Travel-Time LookUp Table (TTLUT)


def get_ttlut(receivers, grid, model, spheric=False):
    """
    Compute Travel-Time LookUpTable for a list of receivers over a 3D grid of possible
    source location for a given 1D velocity model.

    Parameters
    ----------
    receiver: Dataset
        The fiber-optic cable geometry. Must be a 1D dataset with variables
        "longitude", "latitude" and "depth". The dimension can for example be
        "distance" for a fiber-optic cable or "station" for a network of station.
    model: Dataset
        The 1D velocity model in meters per second. Must be a 2D dataset with dimensions
        ("depth", "distance") and variables "p" and/or "s". The chosen resolution will
        affect the quality of the travel-time estimation. The maximal distance must be
        greater than the maximum source-receiver distance. Must be evenly spaced.
    grid: dict
        The 3D grid of possible source locations. Specified as xarray coordinates with
        dimensions ("longitude", "latitude", "depth"). Must be evenly spaced.

    Returns
    -------
    Dataset
        The travel times as a Dataset with one variable per phase ("p" and/or "s") and
        dimensions (..., "longitude", "latitude", "depth").
    """
    (dim,) = receivers.dims
    dst = _get_dst(receivers, grid)
    _, rcv = _to_grid(
        receivers["depth"], xr.DataArray(0.0, {"distance": 0.0}), spheric=spheric
    )
    ttlut = {}
    for phase in model:
        ttgrids = _get_ttgrid(model[phase], rcv)
        tts = []
        for idx, ttgrid in enumerate(ttgrids):
            coords, src = _to_grid(grid["depth"], dst.isel({dim: idx}), spheric=spheric)
            tt = ttgrid(src)
            tt = _from_grid(tt, coords)
            tts.append(tt.astype("float32"))
        tts = xr.concat(tts, dim=dst[dim])
        ttlut[phase] = tts
    return xr.Dataset(ttlut)


def _get_ttgrid(model, rcv):
    gridsize = [_get_step(model[dim]) for dim in ["depth", "distance"]]
    eik = Eikonal2D(
        model.values,
        gridsize=gridsize,
        origin=[model["depth"][0].item(), model["distance"][0].item()],
    )
    ttgrids = eik.solve(rcv)
    return ttgrids


def _get_dst(receivers, grid):
    geod = Geod(ellps="WGS84")
    _, _, dst = geod.inv(
        *xr.broadcast(
            receivers["longitude"],
            receivers["latitude"],
            grid["longitude"],
            grid["latitude"],
        )
    )
    coords = (
        dict(receivers.coords)
        | dict(grid["longitude"].coords)
        | dict(grid["latitude"].coords)
    )
    return xr.DataArray(dst, coords)


def _to_grid(z, x, spheric):
    if spheric:
        z = r0 * np.log(r0 / (r0 - z))
    z, x = xr.broadcast(z, x)
    coords = z.coords
    return coords, np.stack((np.ravel(z), np.ravel(x)), axis=-1)


def _from_grid(t, coords):
    shape = tuple(len(coords[dim]) for dim in coords)
    return xr.DataArray(np.reshape(t, shape), coords)


def _get_step(coord):
    return np.median(np.diff(coord.values))
