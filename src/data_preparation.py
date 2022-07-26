"""Provides functions prepare_data_x.

These functions should take in a dataframe of measurements and return a
PreparedData object.

Note that you can change the input arbitrarily - for example if you want to take
in two dataframes, a dictionary etc. However in this case you will need to edit
the corresponding code in the file prepare_data.py accordingly.

"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.prepared_data import PreparedData
from src.util import (
    CoordDict,
    StanInput,
    get_lognormal_args_from_qs,
    stanify_dict,
)

N_CV_FOLDS = 15
PARAM_NAMES = ["p0", "p1", "p2", "p3", "p4"]
SPECIES_NAMES = ["m", "P1", "P2"]
DIMS = {
    "p": ["parameter"],
    "c": ["species"],
    "conc": ["time", "species"],
    "y": ["observation"],
    "yrep": ["observation"],
    "llik": ["observation"],
}
SEED = 12345


def prepare_data_liepe(
    measurements: pd.DataFrame, param_info: pd.DataFrame
) -> PreparedData:
    """Prepare data following the Liepe et al paper as closely as possible."""
    return PreparedData(
        name="liepe",
        coords=CoordDict(
            {
                "parameter": PARAM_NAMES,
                "observation": measurements.index.tolist(),
                "species": SPECIES_NAMES,
                "time": measurements["time"].tolist(),
            }
        ),
        dims=DIMS,
        measurements=measurements,
        param_info=param_info,
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=get_stan_input_liepe,
    )


def prepare_data_lognormal(
    measurements_raw: pd.DataFrame, param_info_raw: pd.DataFrame
) -> PreparedData:
    """Prepare data following the Liepe et al paper as closely as possible."""
    true_sigma = param_info_raw.query("param == 'sigma'")["true_value"].iloc[0]
    measurements = measurements_raw.copy()
    np.random.seed(SEED)
    measurements["measured_m"] = np.exp(
        np.log(measurements_raw["measured_m"])
        + np.random.normal(0, true_sigma, size=len(measurements_raw))
    )
    return PreparedData(
        name="lognormal",
        coords=CoordDict(
            {
                "parameter": PARAM_NAMES,
                "observation": measurements.index.tolist(),
                "species": SPECIES_NAMES,
                "time": measurements["time"].tolist(),
            }
        ),
        dims=DIMS,
        measurements=measurements,
        param_info=process_param_info_lognormal(param_info_raw),
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=get_stan_input_lognormal,
    )


def process_param_info_lognormal(param_info_raw: pd.DataFrame) -> pd.DataFrame:
    param_info = param_info_raw.copy()
    for i, row in param_info.iterrows():
        (
            param_info.loc[i, "mu"],
            param_info.loc[i, "sigma"],
        ) = get_lognormal_args_from_qs(row["q1pct"], row["q99pct"], 0.01, 0.99)
    return param_info


def get_stan_input_lognormal(
    measurements: pd.DataFrame,
    param_info: pd.DataFrame,
    likelihood: bool,
    train_ix: List[int],
    test_ix: List[int],
) -> StanInput:
    """Turn a processed dataframe into a Stan input."""
    p_info = param_info.loc[lambda df: df["param"].str.startswith("p")]
    sigma_info = param_info.loc[lambda df: df["param"].str.startswith("sigma")]
    return stanify_dict(
        {
            "N": len(measurements),
            "N_train": len(train_ix),
            "N_test": len(test_ix),
            "y_m": measurements["measured_m"],
            "sim_times": measurements["time"],
            "p_loc_and_scale": p_info[["mu", "sigma"]].T,
            "sigma_loc_and_scale": sigma_info[["mu", "sigma"]].T.squeeze(),
            "ix_train": [i + 1 for i in train_ix],
            "ix_test": [i + 1 for i in test_ix],
            "likelihood": int(likelihood),
        }
    )


def get_stan_input_liepe(
    measurements: pd.DataFrame,
    param_info: pd.DataFrame,
    likelihood: bool,
    train_ix: List[int],
    test_ix: List[int],
) -> StanInput:
    """Turn a processed dataframe into a Stan input."""
    unknown_ps = ["p0", "p1", "p2", "p3"]
    p_info = param_info.loc[lambda df: df["param"].isin(unknown_ps)]
    sigma_info = param_info.loc[lambda df: df["param"].str.startswith("sigma")]
    sigma_loc, sigma_scale = get_lognormal_args_from_qs(
        *sigma_info[["q1pct", "q99pct"]].iloc[0], 0.01, 0.99
    )
    return stanify_dict(
        {
            "N": len(measurements),
            "N_train": len(train_ix),
            "N_test": len(test_ix),
            "y_m": measurements["measured_m"],
            "sim_times": measurements["time"],
            "p_upper_bounds": p_info["upper_bound_liepe"],
            "sigma_loc_and_scale": [sigma_loc, sigma_scale],
            "ix_train": [i + 1 for i in train_ix],
            "ix_test": [i + 1 for i in test_ix],
            "likelihood": int(likelihood),
        }
    )


def get_stan_inputs(
    prepared_data: PreparedData,
) -> Tuple[StanInput, StanInput, List[StanInput]]:
    """Get Stan input dictionaries for all modes from a PreparedData object."""
    ix_all = list(range(len(prepared_data.measurements)))
    stan_input_prior, stan_input_posterior = (
        prepared_data.stan_input_function(
            measurements=prepared_data.measurements,
            param_info=prepared_data.param_info,
            likelihood=likelihood,
            train_ix=ix_all,
            test_ix=ix_all,
        )
        for likelihood in (False, True)
    )
    stan_inputs_cv = []
    kf = KFold(prepared_data.number_of_cv_folds, shuffle=True)
    for train, test in kf.split(prepared_data.measurements):
        stan_inputs_cv.append(
            prepared_data.stan_input_function(
                measurements=prepared_data.measurements,
                param_info=prepared_data.param_info,
                likelihood=True,
                train_ix=list(train),
                test_ix=list(test),
            )
        )
    return stan_input_prior, stan_input_posterior, stan_inputs_cv
