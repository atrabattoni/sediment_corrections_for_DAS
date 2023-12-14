## computation

import numpy as np
import xarray as xr

from utils import correlation, multilocalize, multiresidual

multipicks = xr.open_dataarray("data/picks.nc")
count = np.isnan(multipicks).sum(("station", "phase"))
multipicks = multipicks.sel(event=(count >= 100))

ttlut = xr.open_dataarray("/ssd/trabatto/sediment_correction/ttlut_das_geo.nc")
ttlut = ttlut.sel(station=multipicks["station"])  # to remove
ttlut = ttlut.load()

sigma = xr.DataArray([0.1, 0.3, 0.3], coords={"phase": ["Pp", "Ps", "Ss"]})

dt = (multipicks.sel(phase="Ps") - multipicks.sel(phase="Pp")).mean("event")
corr = xr.zeros_like(multipicks.mean("event"))
corr.loc[dict(phase="Ps")] = dt
corr.loc[dict(phase="Ss")] = dt

for niter in range(5):
    print(f"Iteration {niter}")
    locs = multilocalize(ttlut, multipicks - corr, sigma)
    delta = multiresidual(ttlut, multipicks, locs)
    loss = np.square((delta - corr) / sigma).mean().values
    print(f"Locate - loss: {loss}")
    corr = delta.mean("event")
    loss = np.square((delta - corr) / sigma).mean().values
    print(f"Correct - loss: {loss:.3f}")

delta["distance"] = delta["distance"] / 1000
delta = delta.to_dataset("phase")


## figure

import colorcet
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator

plt.style.use("figure.mplstyle")

cmaps = [plt.cm.Blues_r, plt.cm.Oranges_r, plt.cm.Greens_r]

fig, axs = plt.subplot_mosaic(
    [["a", "a", "none"], ["b", "c", "cbar"]],
    width_ratios=[1, 1, 1 / 30],
    figsize=(5.6, 3.2),
)

ax = axs["a"]
ax.axhline(0, color="black")
results = {}
errors = {}
for phase, cmap in zip(delta, cmaps):
    q1 = delta[phase].quantile(0.25, "event")
    q2 = delta[phase].quantile(0.50, "event")
    q3 = delta[phase].quantile(0.75, "event")
    iqr = q1 - q3

    ax.fill_between(q2["distance"], q2, q2, color=cmap(0.0), label=phase, zorder=2)
    ax.fill_between(
        q2["distance"],
        q1,
        q3,
        color=cmap(0.25),
        zorder=1,
        alpha=0.5,
        edgecolor="none",
    )
    ax.fill_between(
        q2["distance"],
        q2 - 1.5 * iqr,
        q2 + 1.5 * iqr,
        color=cmap(0.5),
        zorder=0,
        alpha=0.5,
        edgecolor="none",
    )
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Residual [s]")
    ax.set_xlim(20, 120)
    ax.set_ylim(-2, 2)
    ax.legend(loc="lower center", ncols=3)

    results[phase] = q2
    errors[phase] = (delta[phase] - delta[phase].mean("event")).std().values

for label, phase in [("b", "Ps"), ("c", "Ss")]:
    ax = axs[label]
    sct = ax.scatter(
        x=results["Pp"].values,
        y=results[phase].values,
        c=delta["distance"].values,
        s=3,
        vmin=20,
        vmax=120,
        alpha=1,
        edgecolor="none",
        cmap="cet_CET_R1",
    )
    r, t0 = correlation(
        results["Pp"].values,
        results[phase].values,
        sigma.to_dataset("phase")["Pp"].values,
        sigma.to_dataset("phase")[phase].values,
    )
    ax.plot(
        results["Pp"].values,
        r * results["Pp"].values + t0,
        "k:",
        label=f"y={r:.2f}*x+{t0:.2f}",
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Pp residuals [s]")
    ax.set_ylabel(f"{phase} residuals [s]")
    ax.set_xlim(-0.5, 0.25)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid()
fig.colorbar(sct, cax=axs["cbar"], label="Distance [km]")
axs["cbar"].yaxis.set_major_locator(MultipleLocator(20))
axs["none"].set_axis_off()

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

fig.savefig(f"figs/5_station_correction.jpg")
plt.close(fig)
