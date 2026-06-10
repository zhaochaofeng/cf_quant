"""Signal half-life analysis — Layer 3 of factor evaluation.

Computes the rank IC decay curve across multiple forward-return horizons
and estimates the half-life (in lags) at which predictive power drops to 50%
of the lag-1 baseline.
"""

import numpy as np
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

from utils.logger import LoggerFactory
from utils.stats import WLS

logger = LoggerFactory.get_logger(__name__)


class SignalDecay:
    """Compute IC decay and estimate signal half-life.

    Two methods are provided:
      - calc_half_life (threshold): find first lag where |IC(k)| < 0.5×|IC(1)|.
      - calc_half_life_exp (exponential fit): fit |IC(k)| ≈ |IC(1)|·e^{-λk}
        via log-linear WLS regression.  More robust to noise and avoids NaN
        when IC decays slowly.
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

        baseline= ic_decay.iloc[0]
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
            # 线形插值，计算小数部分
            # Linear interpolation between k_cross-1 and k_cross
            ic_prev = abs(ic_decay[k_cross - 1])
            ic_curr = abs(ic_decay[k_cross])
            # ic(k) crosses threshold between k-1 and k
            # fraction = (ic_prev - threshold) / (ic_prev - ic_curr)
            fraction = (ic_prev - threshold) / (ic_prev - ic_curr)
            half_life = (k_cross - 1) + fraction

        logger.info("Half-life: %.2f lags (baseline IC=%.4f)", half_life, baseline)

        return {"half_life": float(half_life), "ic_decay": ic_decay}

    # ----------------------------------------------------------------
    # Exponential decay fit (alternative method)
    # ----------------------------------------------------------------

    @staticmethod
    def _build_decay_lags(max_lag: int) -> tuple:
        """Generate geometrically spaced lag points up to max_lag.

        Example: max_lag=21 → (1, 2, 4, 8, 16, 21)
        """
        lags = [1]
        while lags[-1] * 2 <= max_lag:
            lags.append(lags[-1] * 2)
        if lags[-1] < max_lag:
            lags.append(max_lag)
        return tuple(lags)

    @staticmethod
    def calc_half_life_exp(
        df: pd.DataFrame,
        factor_col: str,
        ret_prefix: str,
        max_lag: int,
    ) -> dict:
        """Estimate half-life via exponential decay fit (APM Ch13).

        Fits  |IC(k)| ≈ |IC(1)|·e^{-λk}  by log-linear WLS regression
        on geometrically spaced lags, then  half_life = ln(2)/λ.

        Args:
            df: MultiIndex (instrument, datetime) DataFrame.
            factor_col: Column name for factor values.
            ret_prefix: Prefix for forward return columns.
            max_lag: Maximum lag; lags are sampled geometrically.

        Returns:
            dict with keys 'half_life' (float) and 'ic_decay' (Series).
        """
        lags = SignalDecay._build_decay_lags(max_lag)
        ric_means = {}

        for k in lags:
            ret_col = f"{ret_prefix}{k}"
            if ret_col not in df.columns:
                raise ValueError(f"Return column '{ret_col}' not found")
            _, ric = qlib_calc_ic(df[factor_col], df[ret_col])
            ric_means[k] = ric.mean()

        ic_decay = pd.Series(ric_means, name="ric_mean")
        ic_decay.index.name = "lag"

        baseline = ic_decay.iloc[0]
        if pd.isna(baseline) or baseline == 0:
            logger.warning("Baseline IC(1) is zero or NaN")
            return {"half_life": float("nan"), "ic_decay": ic_decay}
        # 由于 log()输入为正数，所以取绝对值。半衰期计算与符号关系不大
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
        logger.info("Half-life (exp fit): %.2f lags (slope=%.4f)", half_life, slope)

        return {"half_life": float(half_life), "ic_decay": ic_decay}
