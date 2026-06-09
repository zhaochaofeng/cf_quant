"""Cross-sectional IC (Information Coefficient) module.

Delegates to qlib's calc_ic() for the core computation, providing a thin
wrapper that operates on MultiIndex (instrument, datetime) DataFrames.
"""

import pandas as pd
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

class CrossSectionalIC:
    """Compute cross-sectional IC between factor exposures and forward returns.

    Delegates the per-date groupby+corr logic to qlib's calc_ic().
    """

    @staticmethod
    def calc_ic(df: pd.DataFrame, factor_col: str, ret_col: str) -> dict:
        """Compute IC and rank IC series from factor and return columns.

        Args:
            df: MultiIndex (instrument, datetime) DataFrame.
            factor_col: Column name for factor values.
            ret_col: Column name for forward return values.

        Returns:
            dict with keys 'ic' (Pearson IC Series) and 'ric' (Spearman rank IC
            Series), each indexed by datetime.
        """
        ic, ric = qlib_calc_ic(df[factor_col], df[ret_col])
        return {"ic": ic, "ric": ric}

    @staticmethod
    def calc_summary(ic_series: pd.Series) -> dict:
        """Compute summary statistics from an IC time series.

        Args:
            ic_series: Time series of IC values (e.g. Pearson or rank IC).

        Returns:
            dict with keys 'ic_mean', 'ic_std', and 'icir' (ICIR = mean / std).
        """
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 and not pd.isna(ic_std) else 0.0
        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
        }
