import numpy as np
import xarray as xr
from scipy.optimize import minimize
from tqdm import tqdm
from xloc import localize


def to_dataframe(picks, sigma):
    sigma = {phase: sigma.sel(phase=phase).item() for phase in sigma["phase"].values}
    picks = picks.to_dataframe("time").reset_index().dropna()
    picks["sigma"] = picks["phase"].map(sigma)
    picks["phase"] = picks["phase"].apply(lambda phase: phase[0])
    picks = picks[["station", "phase", "time", "sigma"]]
    picks = picks.sort_values(["station", "phase"], ignore_index=True)
    return picks


def multilocalize(ttlut, multipicks, sigma, norm=True, return_residuals=False):
    locs = {}
    for event in tqdm(multipicks["event"].values):
        picks = multipicks.sel(event=event, drop=True)
        if return_residuals:
            locs[event] = localize(ttlut, to_dataframe(picks, sigma), norm)
        else:
            locs[event], _ = localize(ttlut, to_dataframe(picks, sigma), norm)
    return locs


def multiresidual(ttlut, multipicks, locs):
    delta = multipicks.copy()
    for event in multipicks["event"].values:
        picks = multipicks.sel(event=event, drop=True)
        delta.loc[dict(event=event)] = get_residuals(ttlut, picks, locs[event])
    return delta


def get_residuals(ttlut, picks, loc):
    tt = ttlut.sel(loc.coords, drop=True)
    delta = picks - loc["time"].values
    for phase in delta["phase"].values:
        delta.loc[dict(phase=phase, station=delta["station"])] -= tt.sel(
            phase=phase[0], station=delta["station"]
        )  # NEW
    return delta


def get_h(dt, vps, vss):
    h = dt / (1 / vss - 1 / vps)
    return h


def get_s(vps, vss, vpb, vsb):
    s = [(1 / vps - 1 / vpb), (1 / vss - 1 / vpb), (1 / vss - 1 / vsb)]
    s = xr.DataArray(s, coords={"phase": ["Pp", "Ps", "Ss"]})
    return s


def solve_vpvs(delta, dt, sigma, vpb, vsb):
    def fun(x):
        vps = x[0]
        vss = x[1]
        h = get_h(dt, vps, vss)
        s = get_s(vps, vss, vpb, vsb)
        return ((delta - h * s) / sigma).var().values

    result = minimize(fun, [2.0, 0.5])
    return np.abs(result.x)


def correlation(x, y, sx, sy):
    """y = a * x + b"""
    mask = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x = x[mask]
    y = y[mask]
    xm = np.mean(x)
    ym = np.mean(y)
    C = np.cov([(x - xm) / sx, (y - ym) / sy])
    evals, evecs = np.linalg.eigh(C)
    a = evecs[1, -1] / evecs[0, -1]
    a *= sy / sx
    b = ym - a * xm
    return a, b
