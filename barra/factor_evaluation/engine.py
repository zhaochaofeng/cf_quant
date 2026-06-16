"""Factor evaluation engine — main orchestration class.

Orchestrates three-layer factor evaluation (APM Ch12-13):

  1. Cross-sectional IC (Information Coefficient)
  2. Stratified return (quantile group returns)
  3. Signal decay (half-life estimation) — alpha factors only
"""

import math
import numpy as np
import pandas as pd
from typing import Optional

from .layer1_ic import CrossSectionalIC
from .layer2_stratified import StratifiedReturn
from .layer3_decay import SignalDecay
from .conf import DEFAULT_N_GROUPS, DEFAULT_MAX_DECAY_LAG
from utils.preprocess import neutralize
from utils.logger import LoggerFactory
from utils import PickleIO, DataFrameIO

logger = LoggerFactory.get_logger(__name__)


def _safe_round(value, precision: int = 6) -> Optional[float]:
    """Round to *precision* decimals, returning None for NaN/inf."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), precision)


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
        ic_periods: tuple = (1,),
        benchmark_close: Optional[pd.Series] = None,
    ):
        """Initialize the factor evaluation engine.

        Args:
            close: MultiIndex (instrument, datetime) Series of stock close prices.
            risk_factors: MultiIndex (instrument, datetime) DataFrame. Each
                column is a risk factor (e.g. LNCAP, BETA). May be None.
            alpha_factors: MultiIndex (instrument, datetime) DataFrame. Each
                column is an alpha factor. May be None.
            ic_periods: Forward-return horizons for IC (Layer 1).
            benchmark_close: MultiIndex (instrument, datetime) Series of
                benchmark (SH000300) close prices. If provided, excess returns
                (stock - benchmark) are computed. Defaults to None (raw returns).

        Raises:
            ValueError: If both risk_factors and alpha_factors are None, or if
                their column names overlap.
        """
        if benchmark_close is None:
            raise ValueError("benchmark_close must not be None")
        
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
        self.ic_periods = ic_periods
        self.benchmark_close = benchmark_close

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        neutralize: bool = False,
        n_groups: int = DEFAULT_N_GROUPS,
        max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
        output: str = './output',
    ) -> dict:
        """Run the full evaluation on all factors.

        Args:
            neutralize: If True, orthogonalize alpha factors against risk
                factors before evaluation.
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
        if not self.ic_periods:
            raise ValueError("ic_periods must not be empty")

        decay_lags = self._build_decay_lags(max_decay_lag, gamma=1.1)
        all_lags = set(self.ic_periods) | set(decay_lags)
        ret_df = self._prepare_forward_returns(all_lags)
        DataFrameIO.write(ret_df, f"{output}/ret_df.parquet")

        result: dict = {}

        # ---- Risk factors: Layer 1 + Layer 2 only (no half-life) ----
        if self.risk_factors is not None and not self.risk_factors.empty:
            result['risk_factors'] = {}
            stratify = StratifiedReturn(n_groups)
            for col in self.risk_factors.columns:
                df = self._build_eval_df(ret_df, self.risk_factors[col], col)
                eval_result = self._evaluate_factor(
                    df, col, self.ic_periods, stratify, decay_lags=None,
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
                    df_raw, col, self.ic_periods, stratify, decay_lags,
                )

                # Neutralized
                if do_neutralize:
                    alpha_neut = self._neutralize_alpha(alpha_series)
                    df_neut = self._build_eval_df(ret_df, alpha_neut, col)
                    alpha_result['neutralized'] = self._evaluate_factor(
                        df_neut, col, self.ic_periods, stratify, decay_lags,
                    )

                result['alpha_factors'][col] = alpha_result

        PickleIO.write(result, f"{output}/result.pkl")
        return result

    # ------------------------------------------------------------------
    # MySQL persistence
    # ------------------------------------------------------------------

    def save_to_mysql(self, result: dict, calc_date: str) -> None:
        """Save evaluation metrics to MySQL ``factor_evaluation`` table.

        Extracts IC/ICIR/RIC/RICIR from Layer 1, long_short/avg_return from
        Layer 2, and half_life from Layer 3 (alpha only).  Uses ``ON
        DUPLICATE KEY UPDATE`` for idempotent upserts based on the
        ``(day, name, type)`` unique key.

        Args:
            result: The dict returned by ``run()``.
            calc_date: Calculation date string (YYYY-MM-DD).
        """

        rows = []
        for name, res in result.get('risk_factors', {}).items():
            row = self._extract_metrics(res, calc_date, name, 'risk')
            rows.append(row)

        for name, res in result.get('alpha_factors', {}).items():
            row = self._extract_metrics(res['neutralized'], calc_date, name, 'alpha')
            rows.append(row)

        if not rows:
            err_msg = f"{calc_date} No rows to save"
            logger.error(err_msg)
            raise ValueError(err_msg)

        from utils import write_to_mysql
        fields = [
            'day', 'name', 'type', 'IC', 'ICIR', 'RIC', 'RICIR', 'long_short',
            'avg_return', 'half_life'
        ]
        unique_key = ['day', 'name', 'type']
        write_to_mysql('factor_evaluation', rows, fields, unique_key, overwrite=True)


    def _extract_metrics(self,
        layer_dict: dict, calc_date: str, name: str, ftype: str,
    ) -> dict:
        """Pull scalar metrics from a single factor's layer results."""
        l1 = layer_dict['layer1'][self.ic_periods[0]]  # 当前仅保存第一个计算周期
        l2 = layer_dict['layer2']
        l3 = layer_dict.get('layer3', {})

        """
            IC / RIC： 用最近日期值作为估计。 
            因为 Ref($close, -k-1) / Ref($close, -k) -1  计算逻辑，calc_date 日期的指标数据不可计算
        """
        return {
            'day': calc_date,
            'name': name,
            'type': ftype,
            'IC': _safe_round(l1.get('ic').dropna().iloc[-1]),
            'ICIR': _safe_round(l1.get('icir')),
            'RIC': _safe_round(l1.get('ric').dropna().iloc[-1]),
            'RICIR': _safe_round(l1.get('ricir')),
            'long_short': _safe_round(l2['long_short'].mean()),
            'avg_return': _safe_round(l2['avg_return'].mean()),
            'half_life': _safe_round(l3.get('half_life')),
        }

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decay_lags(max_lag: int, gamma: float = 1.1) -> tuple:
        """Generate geometrically spaced lag points up to max_lag.
        gamma : 衰减系数（>1）
        Example: max_lag=21 → (1, 2, 4, 8, 16, 21)
        """
        if gamma <= 1:
            raise ValueError(f"gamma must be greater than 1, got {gamma}")
        lags = [1]
        while lags[-1] * gamma <= max_lag:
            lags.append(math.ceil(lags[-1] * gamma))
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
        """Compute excess forward returns (stock return - benchmark return).

           forward_ret_k = stock_ret(k) - benchmark_ret(k)

        where stock_ret(k) = close(t+k+1) / close(t+k) - 1.

        Args:
            lags: Set of lag values for which to generate return columns.

        Returns:
            DataFrame with columns ``forward_ret_{k}`` for each k in lags.
        """
        ret_df = pd.DataFrame(index=self.close.index)
        close_gb = self.close.groupby(level='instrument')
        for k in sorted(lags):
            ret_df[f'forward_ret_{k}'] = close_gb.shift(-k-1) / close_gb.shift(-k) - 1

        # 计算超额收益率
        dates = ret_df.index.get_level_values('datetime')
        for k in sorted(lags):
            bench_ret = self.benchmark_close.shift(-k-1) / self.benchmark_close.shift(-k) -1
            ret_df[f'forward_ret_{k}'] -= bench_ret.reindex(dates).values

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

        # 仅使用第一个 计算周期 ic_periods[0]
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
        result = pd.Series(np.nan, index=alpha_series.index, dtype=float)
        risk = self.risk_factors

        # 过滤 NaN（statsmodels WLS 不自动丢弃）
        valid = alpha_series.notna() & risk.notna().all(axis=1)
        if valid.sum() < 2:
            return result

        result.loc[valid] = neutralize(
            y=alpha_series[valid],
            x=risk.loc[valid],
            weight=1,
            intercept=False,
            level='datetime',
        )
        return result
