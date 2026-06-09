"""Factor evaluation engine — main orchestration class.

Orchestrates three-layer factor evaluation (APM Ch12-13):

  1. Cross-sectional IC (Information Coefficient)
  2. Stratified return (quantile group returns)
  3. Signal decay (half-life estimation)
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

logger = LoggerFactory.get_logger(__name__)


class FactorEvalEngine:
    """Orchestrate three-layer factor evaluation for risk and alpha factors.

    For each factor, computes:
      - Layer 1: Cross-sectional IC (Pearson + Spearman) and ICIR
      - Layer 2: Stratified group returns (quantile portfolios)
      - Layer 3: Signal decay curve and half-life estimation

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
            close: MultiIndex (instrument, datetime) Series of stock close
                prices.
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

    def run(
        self,
        neutralize: bool = False,
        ic_periods: tuple = (1,),
        n_groups: int = DEFAULT_N_GROUPS,
        max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    ) -> dict:
        """Run the full three-layer evaluation on all factors.

        Args:
            neutralize: If True, orthogonalize alpha factors against risk
                factors before evaluation. Ignored when risk_factors is None.
            ic_periods: Forward-return horizons for IC computation
                (Layer 1). Default (1,) = 1-day forward return.
            n_groups: Number of quantile groups for stratified return
                analysis (Layer 2).
            max_decay_lag: Maximum lag for signal decay half-life
                estimation (Layer 3).

        Returns:
            Nested dict with structure::

                {
                    'risk_factors': {
                        '<name>': {
                            'layer1': {period: {...}, ...},
                            'layer2': {...},
                            'layer3': {...},
                        },
                        ...
                    },
                    'alpha_factors': {
                        '<name>': {
                            'raw': {'layer1': {...}, 'layer2': {...}, 'layer3': {...}},
                            'neutralized': {...},  # only if neutralize=True
                        },
                        ...
                    },
                }
        """
        if not ic_periods:
            raise ValueError("ic_periods must not be empty")
        max_lag_needed = max(max(ic_periods), max_decay_lag)
        ret_df = self._prepare_forward_returns(max_lag_needed)

        result: dict = {}

        if self.risk_factors is not None and not self.risk_factors.empty:
            result['risk_factors'] = {}
            stratify = StratifiedReturn(n_groups)
            for col in self.risk_factors.columns:
                df = self._build_eval_df(ret_df, self.risk_factors[col], col)
                result['risk_factors'][col] = self._evaluate_factor(
                    df, col, ic_periods, n_groups, max_decay_lag, stratify
                )

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

                # Raw alpha evaluation
                df_raw = self._build_eval_df(ret_df, alpha_series, col)
                alpha_result['raw'] = self._evaluate_factor(
                    df_raw, col, ic_periods, n_groups, max_decay_lag, stratify
                )

                # Neutralized alpha evaluation
                if do_neutralize:
                    alpha_neut = self._neutralize_alpha(alpha_series)
                    df_neut = self._build_eval_df(ret_df, alpha_neut, col)
                    alpha_result['neutralized'] = self._evaluate_factor(
                        df_neut, col, ic_periods, n_groups, max_decay_lag, stratify
                    )

                result['alpha_factors'][col] = alpha_result

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_forward_returns(self, max_lag: int) -> pd.DataFrame:
        """Compute forward returns for lags 1..max_lag from close prices.

        Args:
            max_lag: Maximum forward-return horizon.

        Returns:
            DataFrame with columns ``forward_ret_1``..``forward_ret_{max_lag}``,
            indexed like ``self.close``.
        """
        ret_df = pd.DataFrame(index=self.close.index)
        close_gb = self.close.groupby(level='instrument')
        for k in range(1, max_lag + 1):
            ret_df[f'forward_ret_{k}'] = close_gb.shift(-k) / self.close - 1
        return ret_df

    @staticmethod
    def _build_eval_df(
        ret_df: pd.DataFrame, factor_series: pd.Series, factor_name: str,
    ) -> pd.DataFrame:
        """Build a DataFrame combining the factor column with forward returns.

        Args:
            ret_df: Forward-return DataFrame (output of _prepare_forward_returns).
            factor_series: Factor values, MultiIndex (instrument, datetime).
            factor_name: Column name to assign to the factor.

        Returns:
            DataFrame with columns ``[factor_name, forward_ret_1, ...]``.
        """
        df = pd.DataFrame({factor_name: factor_series}, index=ret_df.index)
        for col in ret_df.columns:
            df[col] = ret_df[col]
        return df

    def _evaluate_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ic_periods: tuple,
        n_groups: int,
        max_decay_lag: int,
        stratify: StratifiedReturn,
    ) -> dict:
        """Run all three evaluation layers on a single factor.

        Args:
            df: DataFrame with factor_col and forward_ret_* columns.
            factor_col: Column name for the factor.
            ic_periods: Forward-return horizons for IC.
            n_groups: Number of quantile groups for stratified returns.
            max_decay_lag: Maximum lag for signal decay.
            stratify: Pre-instantiated StratifiedReturn object.

        Returns:
            dict with keys 'layer1', 'layer2', 'layer3'.
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
        # Use the first IC period's return column for stratification
        ret_col = f'forward_ret_{ic_periods[0]}'
        layer2_raw = stratify.compute(df, factor_col, ret_col)
        layer2 = {
            'group_returns': layer2_raw['group_returns'],
            'long_short': layer2_raw['long_short'],
        }

        # Layer 3: Signal decay
        layer3 = SignalDecay.calc_half_life(
            df, factor_col, 'forward_ret_', max_decay_lag,
        )

        return {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
        }

    def _neutralize_alpha(self, alpha_series: pd.Series) -> pd.Series:
        """Orthogonalize an alpha factor against risk factors cross-sectionally.

        For each trading date, regress alpha on risk factors via WLS and
        return the residuals (the component of alpha not explained by risk).

        Args:
            alpha_series: Alpha factor values, MultiIndex (instrument, datetime).

        Returns:
            Neutralized alpha Series with the same index as alpha_series.
        """
        alpha_clean = pd.Series(index=alpha_series.index, dtype=float)
        risk = self.risk_factors

        for date in alpha_series.index.get_level_values('datetime').unique():
            mask = alpha_series.index.get_level_values('datetime') == date
            idx = alpha_series[mask].index

            try:
                y_vals = alpha_series.loc[idx].values
                x_vals = risk.loc[idx].values

                # Drop rows where y or any x is NaN
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
