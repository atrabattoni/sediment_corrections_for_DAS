import numpy as np
import pandas as pd
import xarray as xr
from numba import njit


def localize(ttlut, picks, normalize=True):
    """
    Localize an event using a brute-force grid search.

    Parameters
    ----------
    ttlut : DataArray
        Travel-Time LookUp Table. Must have a "station" and a "phase" dimension.
    picks : DataFrame
        Picks. Must have a "station", "phase", "time" and "sigma" columns.

    Returns
    -------
    loc : DataArray
        The retrieved localization. Has a "time" and "residual" field. Coordinates
        gives the localization.
    res : DataArray
        The residuals in seconds.
    """
    is_datetime = np.issubdtype(picks["time"].dtype, np.datetime64)
    picks = picks.copy()
    tref = picks["time"].min()
    picks["time"] = picks["time"] - tref
    if is_datetime:
        picks["time"] = picks["time"] / np.timedelta64(1, "s")
    lut, idx, obs, sig = _to_arrays(ttlut, picks)
    t0, res = _localize(lut, idx, obs, sig, normalize)
    res, loc = _from_arrays(ttlut, t0, res)
    if is_datetime:
        loc["time"] = pd.to_timedelta(loc["time"].values, "s")
    loc["time"] = loc["time"] + tref
    return loc, res


@njit(
    "Tuple((f4[::1], f4[::1]))(f4[:,::1], i2[::1], f4[::1], f4[::1], b1)",
    parallel=True,
)
def _localize(lut, idx, obs, sig, normalize):
    n = len(obs)
    m = lut.shape[-1]
    norm = 0.0
    for i in range(n):
        norm += 1.0 / (sig[i] ** 2)
    t0 = np.zeros(m, dtype="f4")
    for i in range(n):
        t0 += (obs[i] - lut[idx[i]]) / (sig[i] ** 2)
    t0 /= norm
    res = np.zeros(m, dtype="f4")
    for i in range(n):
        res += ((obs[i] - lut[idx[i]] - t0) ** 2) / (sig[i] ** 2)
    if normalize:
        res /= norm
    else:
        res /= n
    res = np.sqrt(res)
    return t0, res


def _to_arrays(ttlut, picks):
    lut = np.ascontiguousarray(ttlut, dtype="f4")
    lut = np.reshape(lut, (ttlut.shape[0] * ttlut.shape[1], -1))
    sta = pd.Categorical(picks["station"], categories=ttlut["station"]).codes
    pha = pd.Categorical(picks["phase"], categories=ttlut["phase"]).codes
    idx = np.ascontiguousarray(sta * ttlut.sizes["phase"] + pha, dtype="i2")
    obs = np.ascontiguousarray(picks["time"], dtype="f4")
    sig = np.ascontiguousarray(picks["sigma"], dtype="f4")
    return lut, idx, obs, sig


def _from_arrays(ttlut, t0, res):
    coords = {dim: ttlut.coords[dim] for dim in ttlut.dims[2:]}
    t0 = xr.DataArray(np.reshape(t0, ttlut.shape[2:]), coords)
    res = xr.DataArray(np.reshape(res, ttlut.shape[2:]), coords)
    idx = res.argmin(...)
    loc = xr.Dataset({"time": t0.isel(idx), "residual": res.isel(idx)})
    return res, loc
