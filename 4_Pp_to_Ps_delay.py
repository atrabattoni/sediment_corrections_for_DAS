import matplotlib.pyplot as plt
import xarray as xr

multipicks = xr.open_dataarray("data/picks.nc")
multipicks["distance"] = multipicks["distance"] / 1000
delay = multipicks.sel(phase="Ps") - multipicks.sel(phase="Pp")
dt = delay.mean("event")
q1 = delay.quantile(0.25, "event")
q2 = delay.quantile(0.50, "event")
q3 = delay.quantile(0.75, "event")
iqr = q1 - q3

plt.style.use("figure.mplstyle")
fig, ax = plt.subplots(figsize=(5.6, 1.6))
cmap = plt.cm.Purples_r
ax.fill_between(q2["distance"], q2, q2, color=cmap(0.0), label="Q2", zorder=2)
ax.fill_between(q2["distance"], q1, q3, color=cmap(0.25), label="Q1/Q3", zorder=1)
ax.fill_between(
    q2["distance"],
    q2 - 1.5 * iqr,
    q2 + 1.5 * iqr,
    color=cmap(0.5),
    label="Q2 +/- 1.5 * (Q3 - Q1)",
    zorder=0,
)
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Delay [s]")
ax.set_xlim(20, 120)
ax.set_ylim(0, 2)
ax.legend(loc="upper center", ncols=3)
fig.savefig("figs/4_Pp_to_Ps_delay.jpg")

dt.to_netcdf("results/delay.nc")
