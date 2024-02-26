# %% Computation

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr
from tqdm import tqdm

from utils import get_h, get_s, multilocalize, multiresidual, solve_vpvs

# parameters
niter = 20
sigma = xr.DataArray([0.1, 0.3, 0.3], coords={"phase": ["Pp", "Ps", "Ss"]})
vps = 2.0
vss = 0.5
vpb = 4.9
vsb = 2.8

# load fiber
fiber = xr.open_dataset("data/fiber.nc")

# load picks
multipicks = xr.open_dataarray("data/picks.nc")

# load ttlut
ttlut = xr.open_dataarray("/ssd/trabatto/sediment_corrections/ttlut.nc").load()

# compute delay
dt = (multipicks.sel(phase="Ps") - multipicks.sel(phase="Pp")).mean("event")

# initialize
h = get_h(dt, vps, vss)
s = get_s(vps, vss, vpb, vsb)
corr = h * s

# iterate
for n in range(1, niter + 1):
    print(f"Iteration {n}")
    locs = multilocalize(ttlut, multipicks - corr, sigma)
    delta = multiresidual(ttlut, multipicks, locs)
    loss = np.square((delta - corr) / sigma).mean().values
    print(f"Locate - loss: {loss}")
    vps, vss = solve_vpvs(delta, dt, sigma, vpb, vsb)
    h = get_h(dt, vps, vss)
    s = get_s(vps, vss, vpb, vsb)
    corr = h * s
    loss = ((delta - corr) / sigma).var().values
    print(
        f"Inverse - loss: {loss:.3f} - Vp: {vps:.3f} - Vs: {vss:.3f} - Vp/Vs: {vps/vss:.3f}"
    )

# result uncertainties
vp = np.linspace(0.0, 4.0, 201)[1:]
vs = np.linspace(0.0, 2.0, 201)[1:]


def process(vpvs):
    vp, vs = vpvs
    if vp == vs:
        corr = 0.0
    else:
        h = get_h(dt, vp, vs)
        s = get_s(vp, vs, vpb, vsb)
        corr = h * s
    t = ((delta - corr) / sigma).mean().values / (
        xr.ones_like(delta - corr) / sigma**2
    ).mean().values
    return (np.square((delta - corr - t) / sigma).mean().values).item()


iterable = itertools.product(vp, vs)
with ProcessPoolExecutor() as executor:
    res = list(
        tqdm(
            executor.map(process, iterable),
            total=len(vp) * len(vs),
            desc="Estimating Uncertainties",
        )
    )
res = np.reshape(res, (len(vp), len(vs)))

# save results
s.to_netcdf("results/s.nc")
h.to_netcdf("results/h.nc")

depth = fiber["depth"].interp(distance=h["distance"])

# format output
delta["distance"] = delta["distance"] / 1000
corr["distance"] = corr["distance"] / 1000
h["distance"] = h["distance"] / 1000
depth = depth / 1000

delta = delta.to_dataset("phase")
corr = corr.to_dataset("phase")
s = s.to_dataset("phase")

# %% Figure

import colorcet
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

plt.style.use("figure.mplstyle")

fig, axs = plt.subplot_mosaic(
    [["a", "a"], ["b", "b"], ["c", "d"]],
    figsize=(5.6, 4.8),
)

ax = axs["a"]
ax.axhline(0, color="black")
cmaps = [plt.cm.Blues_r, plt.cm.Oranges_r, plt.cm.Greens_r]
legend_elements = []
for phase, cmap in zip(delta, cmaps):
    color = cmap(0.0)
    mu = delta[phase].mean("event")
    q1 = delta[phase].quantile(0.25, "event")
    q3 = delta[phase].quantile(0.75, "event")
    ax.fill_between(
        mu["distance"],
        q1,
        q3,
        color=cmap(0.5),
        alpha=0.5,
        edgecolor="none",
    )
    ax.plot(mu["distance"], mu, color=color, ls="--")
    ax.plot(corr["distance"], corr[phase], color=color)
ax.set_ylabel("Correction [s]")
ax.set_xlim(20, 120)
ax.set_ylim(-1, 2)
legend_elements = [
    Line2D([], [], color=cmaps[0](0.0), label="Pp"),
    Line2D([], [], color=cmaps[1](0.0), label="Ps"),
    Line2D([], [], color=cmaps[2](0.0), label="Ss"),
    Line2D([], [], color="black", label="Applied"),
    Line2D([], [], color="black", linestyle="--", label="Required"),
    Patch(facecolor="gray", edgecolor="none", label="Q1/Q3", alpha=0.5),
]
ax.legend(handles=legend_elements, loc="lower center", ncols=6, fontsize=7)
ax.tick_params(labelbottom=False)

ax = axs["b"]
ax.fill_between(h["distance"], 0, depth.values, color="lightblue", alpha=0.5)
ax.fill_between(
    h["distance"], depth.values, depth.values + h.values, color="beige", alpha=0.5
)
ax.fill_between(h["distance"], depth.values + h.values, 4.0, color="gray", alpha=0.3)
ax.axhline(0, color="black", lw=0.75)
ax.plot(h["distance"], depth.values, color="black", lw=0.75)
ax.plot(h["distance"], depth.values + h.values, color="black", lw=0.75)
ax.set_ylim(4, -1)
ax.set_ylabel("Depth [km]")
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlim(20, 120)
ax.set_xlabel("Distance [km]")


for label, phase in [("c", "Ps"), ("d", "Ss")]:
    ax = axs[label]
    sct = ax.scatter(
        x=delta["Pp"].mean("event").values,
        y=delta[phase].mean("event").values,
        c=delta["distance"].values,
        s=2,
        vmin=20,
        vmax=120,
        edgecolor="none",
        cmap="cet_CET_R1",
    )
    r = s[phase] / s["Pp"]
    ax.plot(corr["Pp"], corr[phase], "k", label=f"y={r:.2f}*x")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Pp req. corr. [s]")
    ax.set_ylabel(f"{phase} req. corr. [s]")
    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.50))

cax = axs["d"].inset_axes([1.05, 0.0, 0.05, 1.0])
fig.colorbar(sct, cax=cax, label="Distance [km]")
cax.yaxis.set_major_locator(MultipleLocator(20))
for label in ["a", "b", "c", "d"]:
    axs[label].add_artist(
        AnchoredText(
            f"({label})",
            loc="upper left",
            frameon=False,
            prop=dict(color="black", weight="bold"),
            pad=0.0,
            borderpad=0.2,
        )
    )

fig.savefig(f"figs/6_sediment_correction.pdf")
plt.close(fig)

# %% Figure

fig, ax = plt.subplots(figsize=(3.55, 2.6))
img = ax.pcolormesh(
    vs, vp, res, vmin=0, vmax=2, shading="nearest", cmap="cet_CET_D2", rasterized=True
)
fig.colorbar(img, ax=ax, label="Loss")
ax.contour(vs, vp, res, levels=np.linspace(0, 1, 11), colors="black", linewidths=0.5)
ax.contour(vs, vp, res, levels=[1], colors="black", linewidths=1)
ax.plot(vss, vps, "*", mfc="white", mec="black", mew=0.75, ms=7, label="best model")
ax.set_xlim(0, 2)
ax.set_ylim(0, 4)
ax.set_xlabel("Vs [km/s]")
ax.set_ylabel("Vp [km/s]")
ax.legend()
fig.savefig("figs/7_vp_vs_inversion.pdf")
plt.close(fig)
