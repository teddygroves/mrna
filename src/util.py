"""Some handy python functions."""

from typing import Dict, List, NewType, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

StanInputNumber = Union[float, int]
StanInputList = List[Union[StanInputNumber, "StanInputList"]]
StanInputValue = Union[StanInputNumber, StanInputList]
StanInput = NewType("StanInput", Dict[str, StanInputValue])
CoordDict = NewType("CoordDict", Dict[str, List[str]])


def one_encode(s: pd.Series) -> pd.Series:
    """Replace a series's values with 1-indexed integer factors.

    :param s: a pandas Series that you want to factorise.

    """
    return pd.Series(pd.factorize(s)[0] + 1, index=s.index)


def make_columns_lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame's columns lower case.

    :param df: a pandas DataFrame
    """
    new = df.copy()
    if isinstance(new.columns, pd.MultiIndex):
        new.columns = pd.MultiIndex.from_arrays(
            [
                [c.lower() for c in new.columns.get_level_values(i)]
                for i in range(len(new.columns.levels))
            ]
        )
    else:
        new.columns = pd.Index([c.lower() for c in new.columns])
    return new


def check_is_df(maybe_df) -> pd.DataFrame:
    """Shut up the type checker!"""
    assert isinstance(maybe_df, pd.DataFrame)
    return maybe_df


def stanify_dict(d: Dict) -> StanInput:
    """Make sure a dictionary is a valid Stan input.

    :param d: input dictionary, possibly with wrong types
    """
    out = {}
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValueError(f"key {str(k)} is not a string!")
        elif isinstance(v, pd.Series):
            out[k] = v.to_list()
        elif isinstance(v, pd.DataFrame):
            out[k] = v.values.tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return StanInput(out)


def get_lognormal_args_from_qs(
    x1: float, x2: float, p1: float, p2: float
) -> Tuple[float, float]:
    """Find parameters for a lognormal distribution from two quantiles.

    i.e. get mu and sigma such that if X ~ lognormal(mu, sigma), then pr(X <
    x1) = p1 and pr(X < x2) = p2.

    """
    logx1 = np.log(x1)
    logx2 = np.log(x2)
    denom = norm.ppf(p2) - norm.ppf(p1)
    sigma = (logx2 - logx1) / denom
    mu = (logx1 * norm.ppf(p2) - logx2 * norm.ppf(p1)) / denom
    return mu, sigma
