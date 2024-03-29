# % Computation
import colorcet
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator

from config import ttlut_path

# parameters
sigma = xr.DataArray([0.1, 0.3, 0.3], coords={"phase": ["Pp", "Ps", "Ss"]})

# load picks
multipicks = xr.open_dataarray("data/picks.nc")

# load loc/res
multiloc = pd.read_csv("results/loc.csv", index_col=("event", "correction"))
multires = xr.open_dataset("results/res.nc")

# load corrections
dt = xr.open_dataarray("results/delay.nc")
dt = dt * xr.ones_like(sigma)
dt.loc[dict(phase="Pp")] = 0.0
corr = xr.open_dataarray("results/corr.nc")
h = xr.open_dataarray("results/h.nc")
s = xr.open_dataarray("results/s.nc")
correction = xr.Dataset()
correction["no"] = xr.zeros_like(dt)
correction["dt"] = dt
correction["corr"] = corr
correction["hs"] = h * s

# load fiber
fiber = xr.open_dataset("data/fiber.nc").sel(distance=slice(20_000, 120_000))

# load ttlut
ttlut = xr.open_dataarray(ttlut_path)


for event in multipicks["event"].values:
    if not event == "2021-11-10T01:16:50":  # remove those lines to process all events
        continue

    picks = (
        multipicks.sel(event=event, drop=True)
        .to_dataset("phase")
        .swap_dims(station="distance")
    )
    tref = picks.to_array("phase").min().values.item()
    picks -= tref

    res = multires[event].to_dataset("correction")

    plt.style.use("figure.mplstyle")
    fig, axs = plt.subplots(
        ncols=4,
        nrows=4,
        sharex="row",
        sharey="row",
        figsize=(7.5, 7.0),
        gridspec_kw=dict(height_ratios=[2, 1.25, 2.5, 1.3]),
    )

    title = {
        "no": "No Correction",
        "dt": "Delay Corrections",
        "corr": "Station Corrections",
        "hs": "Sediment Corrections",
    }

    for ax_up, ax_down, kind in zip(axs[0], axs[1], correction):
        t0 = multiloc.loc[event, kind]["time"]
        coords = dict(multiloc.loc[event, kind][["longitude", "latitude", "depth"]])
        tt = ttlut.sel(coords, method="nearest", drop=True).astype("float64")
        tt = tt.to_dataset("phase")
        tt = xr.Dataset({"Pp": tt["P"], "Ps": tt["P"], "Ss": tt["S"]})
        tt = tt.to_array("phase")
        toa = t0 + tt + correction[kind]
        toa = toa.assign_coords(distance=("station", picks["distance"].values))
        toa = toa.to_dataset("phase").swap_dims(station="distance")
        toa -= tref

        for phase in toa:
            toa[phase].plot(ax=ax_up, yincrease=False, color="black", lw=2)

        for phase in picks:
            picks[phase].plot(ax=ax_up, yincrease=False, color="C3", lw=1)

        for idx, phase in enumerate(picks):
            (picks[phase] - toa[phase]).plot(
                ax=ax_down, yincrease=False, lw=4 / 3, zorder=5 - idx
            )
        ax_down.axhline(0, color="black")

        ax_up.set_title(title[kind], fontweight="bold")
        ax_up.set_ylabel("")
        ax_up.set_xlabel("")
        ax_up.tick_params(labelbottom=False)

        ax_down.set_xlabel("Distance [km]")
        ax_down.set_ylabel("")

    ax = axs[0, 0]
    ax.set_xlim(20_000, 120_000)
    ax.xaxis.set_major_locator(MultipleLocator(20_000.0))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1000:g}")
    ax.set_ylabel("Time [s]")
    ax.set_ylim(15, -1)

    ax = axs[1, 0]
    ax.set_xlim(20_000, 120_000)
    ax.xaxis.set_major_locator(MultipleLocator(20_000.0))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1000:g}")
    ax.set_ylabel("Error [s]")
    ax.set_ylim(-1.5, 1.5)

    ax = axs[0, -1]
    ax.plot([], [], color="black", lw=2, label="model")
    ax.legend(loc="lower right")

    ax = axs[0, -2]
    ax.plot([], [], color="C3", lw=4 / 3, label="picks")
    ax.legend(loc="lower right")

    ax = axs[1, 1]
    ax.plot([], [], color="C0", lw=2, label="Pp")
    ax.legend(loc="lower right")

    ax = axs[1, 2]
    ax.plot([], [], color="C1", lw=2, label="Ps")
    ax.legend(loc="lower right")

    ax = axs[1, 3]
    ax.plot([], [], color="C2", lw=2, label="Ss")
    ax.legend(loc="lower right")

    for ax, kind in zip(axs[2], res):
        img = (
            res[kind]
            .min("depth")
            .plot.imshow(
                ax=ax,
                x="longitude",
                y="latitude",
                add_colorbar=False,
                add_labels=False,
                vmin=0,
                vmax=2,
                cmap="cet_CET_D2",
            )
        )
        ctr = (
            res[kind]
            .min("depth")
            .plot.contour(
                ax=ax,
                x="longitude",
                y="latitude",
                add_colorbar=False,
                add_labels=False,
                colors="black",
                levels=[1.0, 2.0],
                linewidths=0.5,
            )
        )
        ax.plot(fiber["longitude"], fiber["latitude"], "k", label="cable")
        coords = dict(multiloc.loc[event, kind][["longitude", "latitude", "depth"]])
        ax.plot(
            coords["longitude"],
            coords["latitude"],
            "*",
            mec="k",
            mfc="w",
            mew=0.75,
            ms=6,
        )
        ax.grid(which="both", color="w", linewidth=0.5, alpha=0.25)
    ax = axs[2, 0]
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(lambda x, _: "")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x}°")
    ax.set_ylabel("Latitude")
    axs[2, -1].legend(loc="lower right")

    for ax, kind in zip(axs[3], res):
        img = (
            res[kind]
            .min("latitude")
            .plot.imshow(
                ax=ax,
                x="longitude",
                y="depth",
                yincrease=False,
                add_colorbar=False,
                add_labels=False,
                vmin=0,
                vmax=2,
                cmap="cet_CET_D2",
            )
        )
        ctr = (
            res[kind]
            .min("latitude")
            .plot.contour(
                ax=ax,
                x="longitude",
                y="depth",
                yincrease=False,
                add_colorbar=False,
                add_labels=False,
                colors="black",
                levels=[1.0, 2.0],
                linewidths=0.5,
            )
        )
        coords = dict(multiloc.loc[event, kind][["longitude", "latitude", "depth"]])
        ax.plot(
            coords["longitude"],
            coords["depth"],
            "*",
            mec="k",
            mfc="w",
            mew=0.75,
            ms=6,
            label="best",
        )
        ax.set_xlabel("Longitude")
        ax.grid(which="both", color="w", linewidth=0.5, alpha=0.25)
    ax = axs[3, 0]
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(10_000.0))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x}°")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1000:g}")
    ax.set_ylabel("Depth [km]")
    axs[3, -1].legend(loc="lower right")

    for label, ax in zip("abcdefghijklmnop", axs.flat):
        ax.add_artist(
            AnchoredText(
                f"({label})",
                loc="upper left",
                frameon=False,
                prop=dict(
                    color="black" if label in "abcdefgh" else "white", weight="bold"
                ),
                pad=0.0,
                borderpad=0.2,
            )
        )

    axs[2, 0].set_xlim(-72.3, -71.1)
    axs[2, 0].set_ylim(-32.9, -31.5)
    axs[3, 0].set_xlim(-72.3, -71.1)
    axs[3, 0].set_ylim(65000, -5000)

    fig.colorbar(
        img,
        ax=axs[3],
        location="bottom",
        orientation="horizontal",
        label="Loss",
        pad=0.2,
        fraction=0.125,
        aspect=30,
    )
    fig.align_ylabels(axs)
    # fig.savefig(f"figs/tmp/{event}.pdf")  # use this line if processing all events
    fig.savefig(f"figs/8_localization.pdf")
    fig.savefig(f"figs/8_localization.jpg")
    plt.close(fig)
