"""Signal half-life analysis — Layer 3 of factor evaluation.

Computes the rank IC decay curve across multiple forward-return horizons
and estimates the half-life (in lags) at which predictive power drops to 50%
of the lag-1 baseline.
"""

import pandas as pd
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

from utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class SignalDecay:
    """Compute IC decay across forward-return horizons and estimate half-life.

    For each lag k from 1 to max_lag, the rank IC between factor values
    and the k-period forward return is computed. The half-life is the lag
    at which |IC(k)| first falls below 50% of |IC(1)|, linearly interpolated
    between adjacent lags.
    """

    @staticmethod
    def calc_half_life(
        df: pd.DataFrame,
        factor_col: str,
        ret_prefix: str,
        max_lag: int,
    ) -> dict:
        """Compute IC decay curve and estimate signal half-life.

        Args:
            df: MultiIndex (instrument, datetime) DataFrame containing
                factor_col and columns named ``{ret_prefix}{k}`` for
                k = 1..max_lag.
            factor_col: Column name for factor values.
            ret_prefix: Prefix for forward return columns (e.g. 'forward_ret_').
            max_lag: Maximum lag (number of periods) to test.

        Returns:
            dict with keys:
                - 'half_life': float, interpolated half-life in lags.
                  NaN if baseline IC is zero/NaN or the signal never decays
                  below 50% of baseline within max_lag.
                - 'ic_decay': pd.Series, index = lag (int), values = mean rank IC.
        """
        if max_lag < 1:
            logger.warning("max_lag=%d < 1, returning NaN half_life", max_lag)
            return {"half_life": float("nan"), "ic_decay": pd.Series(dtype=float)}

        ric_means = {}

        for k in range(1, max_lag + 1):
            ret_col = f"{ret_prefix}{k}"
            if ret_col not in df.columns:
                raise ValueError(f"Return column '{ret_col}' not found in DataFrame")

            _, ric = qlib_calc_ic(df[factor_col], df[ret_col])
            ric_means[k] = ric.mean()

        ic_decay = pd.Series(ric_means, name="ric_mean")
        ic_decay.index.name = "lag"

        baseline = ic_decay.iloc[0]
        baseline_abs = abs(baseline)

        if pd.isna(baseline) or baseline_abs == 0:
            logger.warning("Baseline IC(1) is zero or NaN, cannot compute half-life")
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        threshold = baseline_abs * 0.5

        # Find first lag where absolute IC crosses below threshold
        crossed = abs(ic_decay) < threshold

        if not crossed.any():
            logger.warning(
                "IC decay never crossed below %.4f (50%% of baseline %.4f) "
                "within max_lag=%d",
                threshold, baseline, max_lag,
            )
            return {"half_life": float("nan"), "ic_decay": ic_decay}

        k_cross = crossed.idxmax()  # first index where condition is True
        logger.debug("IC crosses below threshold at lag k=%d", k_cross)

        if k_cross == 1:
            # Already below threshold at lag 1 — unusual but possible
            half_life = 1.0
        else:
            # Linear interpolation between k_cross-1 and k_cross
            ic_prev = abs(ic_decay[k_cross - 1])
            ic_curr = abs(ic_decay[k_cross])
            # ic(k) crosses threshold between k-1 and k
            # fraction = (ic_prev - threshold) / (ic_prev - ic_curr)
            fraction = (ic_prev - threshold) / (ic_prev - ic_curr)
            half_life = (k_cross - 1) + fraction

        logger.info("Half-life: %.2f lags (baseline IC=%.4f)", half_life, baseline)

        return {"half_life": float(half_life), "ic_decay": ic_decay}
