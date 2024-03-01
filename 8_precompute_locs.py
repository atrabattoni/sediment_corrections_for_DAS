# % Computation
from collections import defaultdict

import pandas as pd
import xarray as xr
from tqdm import tqdm

from utils import to_dataframe
from xloc import localize

# parameters
sigma = xr.DataArray([0.1, 0.3, 0.3], coords={"phase": ["Pp", "Ps", "Ss"]})

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

# load picks
multipicks = xr.open_dataarray("data/picks.nc")

# load ttlut
ttlut = xr.open_dataarray("/ssd/trabatto/sediment_corrections/ttlut.nc").load()

multiloc = []
multires = defaultdict(xr.Dataset)
for event in tqdm(multipicks["event"].values):
    ress = xr.Dataset()
    for kind in correction:
        picks = multipicks.sel(event=event, drop=True)
        if kind == "no":
            picks = picks.sel(phase=["Pp", "Ss"])
        picks = to_dataframe(picks - correction[kind], sigma)
        loc, res = localize(ttlut, picks, normalize=False)
        record = {"event": event, "correction": kind} | {
            key: loc[key].values.item() for key in (list(loc.coords) + list(loc))
        }
        multiloc.append(record)
        multires[event][kind] = res

multiloc = pd.DataFrame.from_records(multiloc)
multiloc.to_csv("results/loc.csv", index=False)

multires = xr.Dataset({key: multires[key].to_array("correction") for key in multires})
multires.to_netcdf("results/res.nc")
