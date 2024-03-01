import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdas
import xdas.signal as xp
from matplotlib.offsetbox import AnchoredText

starttime = "2021-11-10T01:16:50"
t_before_in_sec = 2.0

# signal selection & loading
db = xdas.open_database("/data/optodas/post/database.nc")
db = db.sel(time=slice(starttime, None))
db = db.isel(time=slice(125 * 30))
signal = db.to_xarray()

# signal processing
signal = xp.integrate(signal, dim="distance")
signal = xp.decimate(signal, 16, ftype="fir", zero_phase=True, dim="distance")
signal = xp.integrate(signal, dim="time")
signal = xp.decimate(signal, 2, ftype="iir", zero_phase=False, dim="time")
signal = xp.iirfilter(signal, 5.0, "highpass", dim="time")
signal = xp.sliding_mean_removal(signal, 1000.0, dim="distance")
signal *= 1.08e-7

# picks selection & loading
picks = xr.open_dataarray("data/picks.nc").sel(event=starttime, drop=True)
picks = picks.to_dataset("phase")
picks = picks.swap_dims({"station": "distance"})

# relative time
signal["time"] = (signal["time"].values - np.datetime64(0, "s")) / np.timedelta64(
    1, "s"
)
tref = picks.to_array("phase").min().values - t_before_in_sec
picks -= tref
signal["time"] = signal["time"] - tref

# convert to km
picks["distance"] = picks["distance"] / 1000
signal["distance"] = signal["distance"] / 1000

# figure
plt.style.use("figure.mplstyle")
fig, ax = plt.subplots(figsize=(7.5, 5))

# main axis
signal.plot.imshow(
    ax=ax,
    add_colorbar=True,
    add_labels=False,
    norm=mcolors.SymLogNorm(1e-9, vmin=-1e-6, vmax=1e-6),
    cmap="viridis",
    interpolation="none",
    cbar_kwargs=dict(pad=0, extend="neither", label="Displacement [m]", aspect=30),
)
linestyles = {"Pp": "-", "Ps": "-.", "Ss": "--"}
for phase in picks:
    picks[phase].plot(ax=ax, color="C3", linestyle=linestyles[phase], label=phase)
ax.legend(loc="upper right")
ax.set_xlim(20, 120)
ax.set_ylim(25, 0)
ax.set_xlabel("Offset [km]")
ax.set_ylabel("Time [s]")
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

# first inset
x1, x2, y1, y2 = 42.5, 55.0, 6.5, 4
axins = ax.inset_axes(
    [0.01, 0.01, 0.44, 0.34],
    xlim=(x1, x2),
    ylim=(y1, y2),
    xticklabels=[],
    yticklabels=[],
)
signal.plot.imshow(
    ax=axins,
    add_colorbar=False,
    add_labels=False,
    norm=mcolors.SymLogNorm(1e-9, vmin=-1e-7, vmax=1e-7),
    cmap="viridis",
    interpolation="none",
)
rect, lines = ax.indicate_inset_zoom(axins, edgecolor="black")
for line in lines:
    line.set_visible(True)
    line.set_linestyle(":")
axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
axins.set_xlabel("")
axins.set_ylabel("")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.add_artist(
    AnchoredText(
        "(b)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.2,
    )
)

# second inset
x1, x2, y1, y2 = 100, 112.5, 9.5, 7.0
axins = ax.inset_axes(
    [0.55, 0.01, 0.44, 0.34],
    xlim=(x1, x2),
    ylim=(y1, y2),
    xticklabels=[],
    yticklabels=[],
)
signal.plot.imshow(
    ax=axins,
    add_colorbar=False,
    add_labels=False,
    norm=mcolors.SymLogNorm(1e-8, vmin=-1e-6, vmax=1e-6),
    cmap="viridis",
    interpolation="none",
)
rect, lines = ax.indicate_inset_zoom(axins, edgecolor="black")
for line in lines:
    line.set_visible(True)
    line.set_linestyle(":")
axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
axins.set_xlabel("")
axins.set_ylabel("")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.add_artist(
    AnchoredText(
        "(c)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.2,
    )
)

fig.savefig(f"figs/2_signal_and_picks.pdf")
fig.savefig(f"figs/2_signal_and_picks.jpg")
plt.close(fig)
