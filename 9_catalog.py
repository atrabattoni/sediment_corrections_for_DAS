import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText

dataset = xr.open_dataset("data/topo.grd")
fiber = xr.open_dataset("data/fiber.nc")
multiloc = pd.read_csv("results/loc.csv", index_col=("event", "correction"))
multiloc["depth"] /= 1000

plt.style.use("figure.mplstyle")
fig, axes = plt.subplots(
    nrows=2,
    sharex=True,
    figsize=(4.8, 7.15),
    layout="constrained",
    gridspec_kw=dict(height_ratios=[3, 1]),
)

# map
ax = axes[0]
colors_undersea = cmocean.cm.deep_r(np.linspace(0, 1, 256))
colors_land = cmocean.cm.gray(np.linspace(0.2, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = mcolors.LinearSegmentedColormap.from_list("terrain_map", all_colors)
divnorm = mcolors.TwoSlopeNorm(vmin=-6.0, vcenter=0, vmax=6)

ls = mcolors.LightSource(azdeg=45)
data = dataset["z"].data.reshape(dataset["dimension"].values, order="F")[:, ::-1].T
data = data / 1000
profile_dep = -np.nanmean(data, axis=0)
profile_lon = np.linspace(*dataset["x_range"].values, len(profile_dep))
d = 2 * np.pi * 6371 / 360 * dataset["spacing"][1].values
data = ls.shade(data, cmap=terrain_map, blend_mode="overlay", dx=d, dy=d, norm=divnorm)
extent = np.concatenate((dataset["x_range"].values, dataset["y_range"].values))

data[:, :, 3] = 0.5

ax.imshow(data, aspect="auto", origin="lower", extent=extent)

ax.plot(fiber["longitude"], fiber["latitude"], color="black", lw=2, ls="--")
ax.plot(
    fiber["longitude"].sel(distance=slice(20000, 120000)),
    fiber["latitude"].sel(distance=slice(20000, 120000)),
    color="black",
    lw=2,
)


# events
markers = ["o", "s", "H", "D"]
sizes = [23, 20, 26, 17]
labels = ["none", "delay", "station", "sediment"]
colors = ["#ff0000", "#66497d", "#3362a7", "#007ad1"]

for event in multiloc.index.get_level_values(0).unique():
    locs = multiloc.loc[event]
    ax.plot(locs["longitude"], locs["latitude"], c="black", lw=0.5)
    for corr, marker, s, color, in zip(
        multiloc.index.get_level_values(1).unique(), markers, sizes, colors
    ):
        loc = locs.loc[corr]
        sc = ax.scatter(
            [loc["longitude"]],
            [loc["latitude"]],
            color=color,
            ec="black",
            marker=marker,
            s=s,
            linewidth=0.5,
            zorder=3,
        )
for marker, s, color, label in zip(markers, sizes, colors, labels):
    ax.scatter([], [], c=color, ec="black", marker=marker, s=s, label=label)

ax.legend(title="Correction:")
ax.set_aspect("equal", "box")
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(lambda x, pos: f"{abs(x):.1f}°S")
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(lambda x, pos: f"{abs(x):.1f}°W")
ax.set_xlim(-72.3, -71.1)
ax.set_ylim(-32.9, -31.5)

# profile
ax = axes[1]

moho = pd.read_csv("data/moho.csv")
slab = pd.read_csv("data/slab.csv")
ax.plot(moho["longitude"], moho["smoothed_depth"], color="gray", lw=2)
ax.plot(slab["longitude"], slab["smoothed_depth"], color="gray", lw=2)
ax.plot(profile_lon, profile_dep, color="gray", lw=2)

for event in multiloc.index.get_level_values(0).unique():
    locs = multiloc.loc[event]
    ax.plot(locs["longitude"], locs["depth"], c="black", lw=0.5)
    for corr, marker, s, color in zip(
        multiloc.index.get_level_values(1).unique(), markers, sizes, colors
    ):
        # if not corr == "hs":
        #     continue
        loc = locs.loc[corr]
        sc = ax.scatter(
            [loc["longitude"]],
            [loc["depth"]],
            ec="black",
            marker=marker,
            s=s,
            color=color,
            linewidth=0.5,
            zorder=3,
        )
for marker, s, color, label in zip(markers, sizes, colors, labels):
    ax.scatter([], [], c=color, ec="black", marker=marker, s=s, label=label)
ax.set_xlim(-72.3, -71.1)
ax.set_ylim(65, -5)
ax.set_ylabel("Depth [km]")

for ax, label in zip(axes, ["a", "b"]):
    ax.add_artist(
        AnchoredText(
            f"({label})",
            loc="upper left",
            frameon=False,
            prop=dict(color="black", weight="bold"),
            pad=0.0,
            borderpad=0.2,
        )
    )


fig.savefig("figs/9_catalog.pdf", bbox_inches="tight")
plt.close(fig)