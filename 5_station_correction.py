# %% Computation

import numpy as np
import xarray as xr

from config import ttlut_path
from utils import correlation, multilocalize, multiresidual

# parameters
niter = 20
sigma = xr.DataArray([0.1, 0.3, 0.3], coords={"phase": ["Pp", "Ps", "Ss"]})

# load picks
multipicks = xr.open_dataarray("data/picks.nc")

# load ttlut
ttlut = xr.open_dataarray(ttlut_path).load()

# compute delay
dt = (multipicks.sel(phase="Ps") - multipicks.sel(phase="Pp")).mean("event")

# initialize
corr = xr.zeros_like(multipicks.mean("event"))
corr.loc[dict(phase="Ps")] = dt
corr.loc[dict(phase="Ss")] = dt

# iterate
for n in range(1, niter + 1):
    print(f"Iteration {n}")
    locs = multilocalize(ttlut, multipicks - corr, sigma)
    delta = multiresidual(ttlut, multipicks, locs)
    loss = np.square((delta - corr) / sigma).mean().values
    print(f"Locate - loss: {loss}")
    corr = delta.mean("event")
    loss = np.square((delta - corr) / sigma).mean().values
    print(f"Correct - loss: {loss:.3f}")

# save results
corr.to_netcdf("results/corr.nc")

# format output
delta["distance"] = delta["distance"] / 1000
corr["distance"] = corr["distance"] / 1000

delta = delta.to_dataset("phase")
corr = corr.to_dataset("phase")


# %% Figure

import colorcet
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator

plt.style.use("figure.mplstyle")


fig, axs = plt.subplot_mosaic(
    [["a", "a"], ["b", "c"]],
    figsize=(5.6, 3.2),
)

ax = axs["a"]
ax.axhline(0, color="black")
cmaps = [plt.cm.Blues_r, plt.cm.Oranges_r, plt.cm.Greens_r]
for phase, cmap in zip(delta, cmaps):
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
    ax.plot(mu["distance"], mu, color=cmap(0.0), label=phase)
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Correction [s]")
ax.set_xlim(20, 120)
ax.set_ylim(-1, 2)
ax.legend(loc="lower center", fontsize=7, ncols=3)


for label, phase in [("b", "Ps"), ("c", "Ss")]:
    ax = axs[label]
    sct = ax.scatter(
        x=delta["Pp"].mean("event").values,
        y=delta[phase].mean("event").values,
        c=delta["distance"].values,
        s=2,
        vmin=20,
        vmax=120,
        alpha=1,
        edgecolor="none",
        cmap="cet_CET_R1",
    )
    r, t0 = correlation(
        delta["Pp"].values,
        delta[phase].values,
        sigma.to_dataset("phase")["Pp"].values,
        sigma.to_dataset("phase")[phase].values,
    )
    ax.plot(
        delta["Pp"].mean("event"),
        r * delta["Pp"].mean("event") + t0,
        "k",
        label=f"y={r:.2f}*x+{t0:.2f}",
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Pp Correction [s]")
    ax.set_ylabel(f"{phase} Correction [s]")
    ax.set_xlim(-0.5, 0.25)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.50))

cax = axs["c"].inset_axes([1.05, 0.0, 0.05, 1.0])
fig.colorbar(sct, cax=cax, label="Distance [km]")
cax.yaxis.set_major_locator(MultipleLocator(20))
for label in ["a", "b", "c"]:
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

fig.savefig(f"figs/5_station_correction.pdf")
fig.savefig(f"figs/5_station_correction.jpg")
plt.close(fig)
