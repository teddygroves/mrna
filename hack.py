import os

import arviz as az
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

DIR = os.path.join("results", "runs", "liepeetal")

idata_prior, idata_posterior = (
    az.from_json(os.path.join(DIR, f)) for f in ["prior.json", "posterior.json"]
)
f, axes = plt.subplots(2, 3, figsize=[15, 5])
axes = axes
for row, (idata_name, idata) in enumerate(
    zip(["prior", "posterior"], [idata_prior, idata_posterior])
):
    conc_draws = (
        idata.posterior["conc"]
        .to_series()
        .unstack(["species", "time"])
        .sort_index(axis=1)
    )
    species = conc_draws.columns.levels[0].values
    times = conc_draws.columns.levels[1].values
    for (chain, draw), _ in conc_draws.sample(15).iterrows():
        for col, s in enumerate(species):
            t = conc_draws.loc[(chain, draw), s]
            ax = axes[row, col]
            ax.plot(t.index, t.values, linewidth=0.1, label=s, color="black")
            ax.set_title(f"{species[col].capitalize()}: {idata_name}")
    axes[row,-1].scatter(times, idata.observed_data["y_m"].values, color="red")
plt.tight_layout()
plt.show()
