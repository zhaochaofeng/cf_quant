"""Factor evaluation engine — main orchestration class.

Orchestrates three-layer factor evaluation (APM Ch12-13):

  1. Cross-sectional IC (Information Coefficient)
  2. Stratified return (quantile group returns)
  3. Signal decay (half-life estimation) — alpha factors only
"""

import numpy as np
import pandas as pd
from typing import Optional

from .layer1_ic import CrossSectionalIC
from .layer2_stratified import StratifiedReturn
from .layer3_decay import SignalDecay
from .config import DEFAULT_N_GROUPS, DEFAULT_MAX_DECAY_LAG
from utils.preprocess import neutralize
from utils.logger import LoggerFactory
from utils import PickleIO

logger = LoggerFactory.get_logger(__name__)


class FactorEvalEngine:
    """Orchestrate three-layer factor evaluation for risk and alpha factors.

    Layer 3 (half-life) is only computed for alpha factors — risk factors
    are structural exposures that do not decay over time.

    Alpha factors can optionally be orthogonalized against risk factors
    before evaluation (neutralize=True).
    """

    def __init__(
        self,
        close: pd.Series,
        risk_factors: Optional[pd.DataFrame] = None,
        alpha_factors: Optional[pd.DataFrame] = None,
    ):
        """Initialize the factor evaluation engine.

        Args:
            close: MultiIndex (instrument, datetime) Series of stock close prices.
            risk_factors: MultiIndex (instrument, datetime) DataFrame. Each
                column is a risk factor (e.g. LNCAP, BETA). May be None.
            alpha_factors: MultiIndex (instrument, datetime) DataFrame. Each
                column is an alpha factor. May be None.

        Raises:
            ValueError: If both risk_factors and alpha_factors are None, or if
                their column names overlap.
        """
        if risk_factors is None and alpha_factors is None:
            raise ValueError(
                "At least one of risk_factors or alpha_factors must be provided."
            )
        if risk_factors is not None and alpha_factors is not None:
            overlap = set(risk_factors.columns) & set(alpha_factors.columns)
            if overlap:
                raise ValueError(
                    f"Column name overlap between risk_factors and alpha_factors: "
                    f"{overlap}"
                )

        self.close = close
        self.risk_factors = risk_factors
        self.alpha_factors = alpha_factors

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        neutralize: bool = False,
        ic_periods: tuple = (1,),
        n_groups: int = DEFAULT_N_GROUPS,
        max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
        output: str = './output',
    ) -> dict:
        """Run the full evaluation on all factors.

        Args:
            neutralize: If True, orthogonalize alpha factors against risk
                factors before evaluation.
            ic_periods: Forward-return horizons for IC (Layer 1).
            n_groups: Number of quantile groups for stratified return (Layer 2).
            max_decay_lag: Maximum lag for half-life estimation (Layer 3).
            output: Directory path for intermediate PickleIO results.

        Returns:
            Nested dict::

                {
                    'risk_factors': {
                        '<name>': {'layer1': {...}, 'layer2': {...}},
                        ...
                    },
                    'alpha_factors': {
                        '<name>': {
                            'raw': {'layer1':{...}, 'layer2':{...}, 'layer3':{...}},
                            'neutralized': {...},  # only if neutralize=True
                        },
                        ...
                    },
                }
        """
        if not ic_periods:
            raise ValueError("ic_periods must not be empty")

        decay_lags = self._build_decay_lags(max_decay_lag)
        all_lags = set(ic_periods) | set(decay_lags)
        ret_df = self._prepare_forward_returns(all_lags)

        result: dict = {}

        # ---- Risk factors: Layer 1 + Layer 2 only (no half-life) ----
        if self.risk_factors is not None and not self.risk_factors.empty:
            result['risk_factors'] = {}
            stratify = StratifiedReturn(n_groups)
            for col in self.risk_factors.columns:
                df = self._build_eval_df(ret_df, self.risk_factors[col], col)
                eval_result = self._evaluate_factor(
                    df, col, ic_periods, stratify, decay_lags=None,
                )
                result['risk_factors'][col] = eval_result

        # ---- Alpha factors: full three-layer ----
        if self.alpha_factors is not None and not self.alpha_factors.empty:
            result['alpha_factors'] = {}
            stratify = StratifiedReturn(n_groups)
            do_neutralize = (
                neutralize and self.risk_factors is not None
                and not self.risk_factors.empty
            )
            for col in self.alpha_factors.columns:
                alpha_result = {}
                alpha_series = self.alpha_factors[col]

                # Raw
                df_raw = self._build_eval_df(ret_df, alpha_series, col)
                alpha_result['raw'] = self._evaluate_factor(
                    df_raw, col, ic_periods, stratify, decay_lags,
                )

                # Neutralized
                if do_neutralize:
                    alpha_neut = self._neutralize_alpha(alpha_series)
                    df_neut = self._build_eval_df(ret_df, alpha_neut, col)
                    alpha_result['neutralized'] = self._evaluate_factor(
                        df_neut, col, ic_periods, stratify, decay_lags,
                    )

                result['alpha_factors'][col] = alpha_result

        PickleIO.write(result, f"{output}/result.pkl")
        return result

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

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
    def _build_eval_df(
        ret_df: pd.DataFrame, factor_series: pd.Series, factor_name: str,
    ) -> pd.DataFrame:
        """Combine factor column with forward-return columns."""
        df = pd.DataFrame({factor_name: factor_series}, index=ret_df.index)
        for col in ret_df.columns:
            df[col] = ret_df[col]
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_forward_returns(self, lags: set) -> pd.DataFrame:
        """Compute single-day marginal forward returns for requested lags.

           forward_ret_k = close(t+k+1) / close(t+k) - 1

        Args:
            lags: Set of lag values for which to generate return columns.

        Returns:
            DataFrame with columns ``forward_ret_{k}`` for each k in lags.
        """
        ret_df = pd.DataFrame(index=self.close.index)
        close_gb = self.close.groupby(level='instrument')
        for k in sorted(lags):
            ret_df[f'forward_ret_{k}'] = close_gb.shift(-k-1) / close_gb.shift(-k) - 1
        return ret_df

    def _evaluate_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ic_periods: tuple,
        stratify: StratifiedReturn,
        decay_lags: Optional[tuple] = None,
    ) -> dict:
        """Run evaluation layers on a single factor.

        Args:
            decay_lags: Pre-computed geometric lags for Layer 3.
                If None, Layer 3 is skipped (for risk factors).
        """
        # Layer 1: Cross-sectional IC
        layer1 = {}
        for period in ic_periods:
            ret_col = f'forward_ret_{period}'
            ic_result = CrossSectionalIC.calc_ic(df, factor_col, ret_col)
            ic_summary = CrossSectionalIC.calc_summary(ic_result['ic'])
            ric_summary = CrossSectionalIC.calc_summary(ic_result['ric'])
            layer1[period] = {
                'ic': ic_result['ic'],
                'ric': ic_result['ric'],
                'ic_mean': ic_summary['ic_mean'],
                'ic_std': ic_summary['ic_std'],
                'icir': ic_summary['icir'],
                'ric_mean': ric_summary['ic_mean'],
                'ric_std': ric_summary['ic_std'],
                'ricir': ric_summary['icir'],
            }

        # Layer 2: Stratified return
        ret_col = f'forward_ret_{ic_periods[0]}'
        layer2 = stratify.compute(df, factor_col, ret_col)

        # Layer 3: Signal decay (alpha only)
        if decay_lags is not None:
            layer3 = SignalDecay.calc_half_life(
                df, factor_col, 'forward_ret_', decay_lags,
            )
            return {'layer1': layer1, 'layer2': layer2, 'layer3': layer3}

        return {'layer1': layer1, 'layer2': layer2}

    def _neutralize_alpha(self, alpha_series: pd.Series) -> pd.Series:
        """Orthogonalize alpha against risk factors cross-sectionally.

        For each date, regress alpha on risk factors via WLS (no intercept)
        and return the residuals.
        """
        alpha_clean = pd.Series(index=alpha_series.index, dtype=float)
        risk = self.risk_factors

        for date in alpha_series.index.get_level_values('datetime').unique():
            mask = alpha_series.index.get_level_values('datetime') == date
            idx = alpha_series[mask].index

            try:
                y_vals = alpha_series.loc[idx].values
                x_vals = risk.loc[idx].values

                valid = ~np.isnan(y_vals) & ~np.isnan(x_vals).any(axis=1)
                if valid.sum() < 2:
                    logger.debug(
                        "Skipping date %s: insufficient valid observations (%d)",
                        date, valid.sum(),
                    )
                    alpha_clean.loc[idx] = np.nan
                    continue

                residuals = neutralize(
                    y=y_vals[valid],
                    x=x_vals[valid],
                    weight=1,
                    intercept=False,
                )
                alpha_clean.loc[idx[valid]] = residuals

            except Exception:
                logger.warning(
                    "Neutralization failed for date %s, factor %s",
                    date, alpha_series.name,
                )
                alpha_clean.loc[idx] = np.nan

        return alpha_clean
