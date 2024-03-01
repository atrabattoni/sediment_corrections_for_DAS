import cmocean
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.offsetbox import AnchoredText

santiago = dict(latitude=-33.4733, longitude=-70.6503)
concon = dict(latitude=-32.9597, longitude=-71.5155)
laserena = dict(latitude=-29.9448, longitude=-71.2546)
psr = dict(latitude=-32.30, longitude=-72.00)
vb = dict(latitude=-32.80, longitude=-72.05)

dataset = xr.open_dataset("data/topo.grd")
fiber = xr.open_dataset("data/fiber.nc")

plt.style.use("figure.mplstyle")
fig, axes = plt.subplots(
    nrows=2, figsize=(3.55, 3.70), gridspec_kw=dict(height_ratios=[6, 1])
)

ax = axes[0]
colors_undersea = cmocean.cm.deep_r(np.linspace(0, 1, 256))
colors_land = cmocean.cm.gray(np.linspace(0.2, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = mcolors.LinearSegmentedColormap.from_list("terrain_map", all_colors)
divnorm = mcolors.TwoSlopeNorm(vmin=-6.0, vcenter=0, vmax=6)

ls = mcolors.LightSource(azdeg=45)
data = dataset["z"].data.reshape(dataset["dimension"].values, order="F")[:, ::-1].T
data = data / 1000
d = 2 * np.pi * 6371 / 360 * dataset["spacing"][1].values
data = ls.shade(data, cmap=terrain_map, blend_mode="overlay", dx=d, dy=d, norm=divnorm)
extent = np.concatenate((dataset["x_range"].values, dataset["y_range"].values))

ax.imshow(data, aspect="auto", origin="lower", extent=extent)
fig.colorbar(
    mcm.ScalarMappable(norm=divnorm, cmap=terrain_map),
    ax=ax,
    orientation="vertical",
    extend="neither",
    label="Elevation [km]",
    ticks=mticker.MultipleLocator(2.0),
)
ax.plot(
    fiber["longitude"].sel(distance=slice(None, 150_000)),
    fiber["latitude"].sel(distance=slice(None, 150_000)),
    color="C3",
    lw=1,
    ls=":",
)
ax.plot(
    fiber["longitude"].sel(distance=slice(20_000, 120_000)),
    fiber["latitude"].sel(distance=slice(20_000, 120_000)),
    color="C3",
    lw=1,
)
for distance in [20_000, 40_000, 60_000, 80_000, 100_000, 120_000]:
    x = fiber["longitude"].sel(distance=distance, method="nearest")
    y = fiber["latitude"].sel(distance=distance, method="nearest")
    ax.plot(x, y, ".C3", ms=2)
ax.scatter(
    santiago["longitude"], santiago["latitude"], s=8, marker="s", fc="w", ec="k", lw=0.3
)
ax.scatter(
    concon["longitude"], concon["latitude"], s=8, marker="s", fc="w", ec="k", lw=0.3
)
ax.text(
    santiago["longitude"] - 0.03,
    santiago["latitude"] - 0.03,
    "Santiago",
    color="w",
    ha="right",
    va="top",
    clip_on=True,
)
ax.text(
    concon["longitude"] + 0.04,
    concon["latitude"] - 0.04,
    "Concón",
    color="w",
    ha="left",
    va="top",
    clip_on=True,
)
ax.text(
    vb["longitude"],
    vb["latitude"],
    "VB",
    color="k",
    ha="center",
    va="center",
    clip_on=True,
)
ax.text(
    psr["longitude"],
    psr["latitude"],
    "PSR",
    rotation=35,
    color="k",
    ha="center",
    va="center",
    clip_on=True,
)
ax.set_aspect("equal", "box")
ax.set_xlim(-73.5, -70.5)
ax.set_ylim(-34, -31)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.yaxis.set_major_formatter(lambda x, pos: f"{abs(x):.0f}°S")
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.xaxis.set_major_formatter(lambda x, pos: f"{abs(x):.0f}°W")
ax.add_artist(
    AnchoredText(
        "(a)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.2,
    )
)

ax = axes[1]
ax.fill_between(fiber["distance"] / 1000, 0.0, 3.0, color="skyblue")
ax.fill_between(fiber["distance"] / 1000, fiber["depth"] / 1000, 3.0, color="gray")
ax.plot(fiber["distance"] / 1000, fiber["depth"] / 1000, color="C3", ls=":")
ax.plot(
    fiber["distance"].sel(distance=slice(20_000, 120_000)) / 1000,
    fiber["depth"].sel(distance=slice(20_000, 120_000)) / 1000,
    color="C3",
)
ax.set_xlim(0, 150)
ax.set_ylim(3, 0)
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Depth [km]")
ax.xaxis.set_major_locator(mticker.MultipleLocator(20.0))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.grid(color="white", linestyle=":")
ax.add_artist(
    AnchoredText(
        "(b)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.2,
    )
)

fig.savefig("figs/1_map.pdf")
fig.savefig("figs/1_map.jpg")
plt.close(fig)
