"""Signal half-life estimation via exponential decay fit — Layer 3 of factor evaluation.

Fits  |IC(k)| ≈ |IC(1)|·e^{-λk}  by log-linear WLS regression, then
half_life = ln(2)/λ  (APM Ch13).
"""

import numpy as np
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

from utils.logger import LoggerFactory
from utils.stats import WLS

logger = LoggerFactory.get_logger(__name__)


class SignalDecay:
    """Estimate signal half-life via exponential decay fit."""

    @staticmethod
    def calc_half_life(
        df: pd.DataFrame,
        factor_col: str,
        ret_prefix: str,
        lags: tuple,
    ) -> dict:
        """Estimate half-life by fitting |IC(k)| ≈ |IC(1)|·e^{-λk}.

        Args:
            df: MultiIndex (instrument, datetime) DataFrame.
            factor_col: Column name for factor values.
            ret_prefix: Prefix for forward return columns (e.g. 'forward_ret_').
            lags: Tuple of lag values at which to compute IC (geometrically
                spaced, pre-computed by the caller).

        Returns:
            dict with keys 'half_life' (float) and 'ic_decay' (Series).
        """
        if len(lags) < 3:
            logger.warning("Need at least 3 lags for exponential fit, got %d", len(lags))
            return {"half_life": float("nan"), "ic_decay": pd.Series(dtype=float)}

        ric_means = {}
        for k in lags:
            ret_col = f"{ret_prefix}{k}"
            if ret_col not in df.columns:
                raise ValueError(f"Return column '{ret_col}' not found in DataFrame")
            _, ric = qlib_calc_ic(df[factor_col], df[ret_col])
            ric_means[k] = ric.mean()

        ic_decay = pd.Series(ric_means, name="ric_mean")
        ic_decay.index.name = "lag"

        baseline = ic_decay.iloc[0]
        if pd.isna(baseline) or baseline == 0:
            logger.warning("Baseline IC(1) is zero or NaN")
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        abs_ic = np.abs(ic_decay.values)
        lags_arr = np.array(lags, dtype=float)

        valid = abs_ic > 0
        if valid.sum() < 3:
            logger.warning("Fewer than 3 valid points (|IC| > 0), cannot fit")
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        log_ic = np.log(abs_ic[valid])
        k_vals = lags_arr[valid]

        try:
            b, _, _ = WLS(y=log_ic, X=k_vals.reshape(-1, 1),
                          intercept=True, weight=1)
            slope = b[0]
        except Exception:
            logger.warning("WLS fit failed for half-life estimation")
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        if slope >= 0:
            logger.warning("Slope b=%.4f >= 0, IC not decaying", slope)
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        half_life = -np.log(2) / slope

        return {"half_life": float(half_life), "ic_decay": ic_decay}
