"""Read the data in RAW_DIR and save prepared data to PREPARED_DIR."""

import json
import os

import pandas as pd
from cmdstanpy import write_stan_json

from src.data_preparation import (
    get_stan_inputs,
    prepare_data_liepe,
    prepare_data_lognormal,
)
from src.util import check_is_df

RAW_DIR = os.path.join("data", "raw")
RAW_DATA_FILES = {
    "raw_measurements": os.path.join(RAW_DIR, "raw_measurements.csv"),
    "raw_param_info": os.path.join(RAW_DIR, "param_info.csv"),
}

PREPARED_DIR = os.path.join("data", "prepared")


def main():
    """Save prepared data in the PREPARED_DATA_DIR."""
    print("Reading raw data...")
    raw_data = {
        k: check_is_df(pd.read_csv(v, index_col=None))
        for k, v in RAW_DATA_FILES.items()
    }
    print("Preparing data...")
    for data_prep_function in [
        prepare_data_liepe,
        prepare_data_lognormal,
    ]:
        prepared_data = data_prep_function(
            raw_data["raw_measurements"], raw_data["raw_param_info"]
        )
        output_dir = os.path.join(PREPARED_DIR, prepared_data.name)
        cv_dir = os.path.join(output_dir, "stan_inputs_cv")
        measurements_file = os.path.join(output_dir, "measurements.csv")
        param_info_file = os.path.join(output_dir, "param_info.csv")
        input_file_prior, input_file_posterior = (
            os.path.join(output_dir, f"stan_input_{s}.json")
            for s in ["prior", "posterior"]
        )
        for d in [PREPARED_DIR, output_dir, cv_dir]:
            if not os.path.exists(d):
                os.mkdir(d)
        si_prior, si_posterior, sis_cv = get_stan_inputs(prepared_data)
        prepared_data.measurements.to_csv(measurements_file)
        prepared_data.param_info.to_csv(param_info_file)
        write_stan_json(input_file_posterior, si_posterior)
        write_stan_json(input_file_prior, si_prior)
        for i, si in enumerate(sis_cv):
            f = os.path.join(cv_dir, f"split_{str(i)}.json")
            write_stan_json(f, si)
        with open(os.path.join(output_dir, "coords.json"), "w") as f:
            json.dump(prepared_data.coords, f)
        with open(os.path.join(output_dir, "dims.json"), "w") as f:
            json.dump(prepared_data.dims, f)
        print(f"Finished preparing data folder {output_dir}.")


if __name__ == "__main__":
    main()
