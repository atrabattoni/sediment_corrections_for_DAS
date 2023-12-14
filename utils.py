import numpy as np

# import xarray as xr
# from scipy.optimize import basinhopping, minimize
from tqdm import tqdm
from xloc import localize

# def get_u(r):
#     return xr.DataArray([1.0 / r, 1.0, 1.0], coords={"phase": ["Pp", "Ps", "Ss"]})


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


# def solve_r(delta, dt, sigma):
#     def fun(x):
#         t = x[0]
#         r = x[1]
#         u = get_u(r)
#         v = dt * r / (r - 1)
#         return np.square((delta - u * v - t) / sigma).mean().values

#     result = minimize(fun, [0.0, np.sqrt(3)])
#     return result.x


# def get_h(dt, vps, vss):
#     h = dt / (1 / vss - 1 / vps)
#     return h


# def get_s(vps, vss, vpb, vsb):
#     s = [(1 / vps - 1 / vpb), (1 / vss - 1 / vpb), (1 / vss - 1 / vsb)]
#     s = xr.DataArray(s, coords={"phase": ["Pp", "Ps", "Ss"]})
#     return s


# def solve_vpvs(delta, dt, sigma, vpb, vsb):
#     def fun(x):
#         t = x[0]
#         vps = x[1]
#         vss = x[2]
#         h = get_h(dt, vps, vss)
#         s = get_s(vps, vss, vpb, vsb)
#         return np.square((delta - h * s - t) / sigma).mean().values

#     result = minimize(fun, [0.0, 2.0, 0.5])
#     return result.x


# def solve_multi_vpvs(delta, dt, sigma, vpb, vsb, nsection):
#     deltas = np.array_split(delta, nsection)
#     dts = np.array_split(dt, nsection)

#     def fun(x):
#         t = x[0]
#         vps = x[1 : nsection + 1]
#         vss = x[nsection + 1 : 2 * nsection + 1]
#         hs = [None] * nsection
#         ss = [None] * nsection
#         corrs = [None] * nsection
#         for k in range(nsection):
#             hs[k] = get_h(dts[k], vps[k], vss[k])
#             ss[k] = get_s(vps[k], vss[k], vpb, vsb)
#             corrs[k] = hs[k] * ss[k]
#         corr = xr.concat(corrs, "station")
#         return np.square((delta - corr - t) / sigma).mean().values

#     result = minimize(fun, [0.0] + [2.0] * nsection + [0.5] * nsection)
#     return (
#         result.x[0],
#         result.x[1 : nsection + 1],
#         result.x[nsection + 1 : 2 * nsection + 1],
#     )


# def solve_multi_vs(delta, dt, sigma, vpb, vsb, nsection, vps):
#     deltas = np.array_split(delta, nsection)
#     dts = np.array_split(dt, nsection)
#     vps = [vps] * nsection

#     def fun(x):
#         t = x[0]
#         vss = x[1:]
#         hs = [None] * nsection
#         ss = [None] * nsection
#         corrs = [None] * nsection
#         for k in range(nsection):
#             hs[k] = get_h(dts[k], vps[k], vss[k])
#             ss[k] = get_s(vps[k], vss[k], vpb, vsb)
#             corrs[k] = hs[k] * ss[k]
#         corr = xr.concat(corrs, "station")
#         return np.square((delta - corr - t) / sigma).mean().values

#     result = basinhopping(fun, [0.0] + [0.5] * nsection)
#     return result.x[0], vps, result.x[1:]


# def get_uv(r, dt):
#     indices = np.linspace(0, len(dt), len(r))
#     r = np.interp(np.arange(len(dt)), indices, r)
#     r = xr.DataArray(r, {"station": dt["station"]})
#     coeff = r / (r - 1)
#     coeff[np.isnan(coeff)] = 1.0
#     v = dt * coeff
#     uv = xr.Dataset()
#     uv["Pp"] = v / r
#     uv["Ps"] = v
#     uv["Ss"] = v
#     uv = uv.to_array("phase")
#     return uv


# def solve_tr(t, r, delta, dt, sigma):
#     def fun(x):
#         t = x[0]
#         r = x[1:]
#         uv = get_uv(r, dt)
#         return np.square((delta - uv - t) / sigma).mean().values

#     result = minimize(fun, np.concatenate(([t], r)))
#     return result.x[0], result.x[1:]


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
