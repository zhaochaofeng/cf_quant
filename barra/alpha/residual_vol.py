"""
残差波动率估计模块 - 老股票历史std + 新股回归预测
"""
import pandas as pd
import numpy as np

from .config import NEW_STOCK_MIN_DAYS, VOL_ROLLING_WINDOW
from utils import WLS, LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ResidualVolEstimator:
    """残差波动率估计器

    老股票：用历史残差收益率的标准差
    新股：用行业+log(市值)回归模型预测
    """

    def __init__(
        self,
        vol_window: int = NEW_STOCK_MIN_DAYS,
        vol_rolling: int = VOL_ROLLING_WINDOW
    ):
        """初始化

        Args:
            vol_window: 老股判定阈值（交易日数），低于此值视为新股
            vol_rolling: 波动率滚动窗口（交易日数），取最近N天计算std
        """
        self.vol_window = vol_window
        self.vol_rolling = vol_rolling

    def estimate_all(
        self,
        residuals: pd.DataFrame,
        industry_df: pd.DataFrame,
        market_cap_df: pd.DataFrame,
        as_of_date: str
    ) -> pd.Series:
        """估计所有股票的残差波动率

        Args:
            residuals: 残差收益率，MultiIndex(instrument, datetime), column='residual'
            industry_df: 行业数据，MultiIndex(instrument, datetime), column='industry_code'
            market_cap_df: 市值数据，MultiIndex(instrument, datetime), column='circ_mv'
            as_of_date: 计算截止日期

        Returns:
            Series(instrument -> omega)
        """
        omega_hist = self._compute_historical_vol(residuals, as_of_date)
        old_stocks, new_stocks = self._classify_stock_age(residuals, as_of_date)

        omega_old = omega_hist.loc[omega_hist.index.isin(old_stocks)]
        logger.info(f'老股票 {len(old_stocks)} 只, 新股 {len(new_stocks)} 只')

        if not new_stocks:
            return omega_old

        # 为新股预测omega
        # 取as_of_date当天的行业和市值截面数据
        as_of_ts = pd.Timestamp(as_of_date)
        try:
            ind_cross = industry_df.xs(as_of_ts, level='datetime')
            mv_cross = market_cap_df.xs(as_of_ts, level='datetime')
        except KeyError:
            # 若as_of_date无数据，取最近日期
            dates = industry_df.index.get_level_values('datetime').unique()
            nearest = dates[dates <= as_of_ts].max()
            ind_cross = industry_df.xs(nearest, level='datetime')
            mv_cross = market_cap_df.xs(nearest, level='datetime')

        # 拟合回归模型
        old_in_ind = ind_cross.loc[ind_cross.index.isin(old_stocks)]
        old_in_mv = mv_cross.loc[mv_cross.index.isin(old_stocks)]
        common_old = omega_old.index.intersection(old_in_ind.index).intersection(old_in_mv.index)

        if len(common_old) < 30:
            logger.warning(f'老股票样本不足({len(common_old)})，新股omega取中位数')
            median_omega = omega_old.median()
            omega_new = pd.Series(median_omega, index=new_stocks, name='omega')
            return pd.concat([omega_old, omega_new]).sort_index()

        params, intercept = self._fit_regression(
            omega_old.loc[common_old],
            old_in_ind.loc[common_old],
            old_in_mv.loc[common_old]
        )

        # 预测新股omega
        new_in_ind = ind_cross.loc[ind_cross.index.isin(new_stocks)]
        new_in_mv = mv_cross.loc[mv_cross.index.isin(new_stocks)]
        common_new = new_in_ind.index.intersection(new_in_mv.index)

        if common_new.empty:
            logger.warning('新股无行业/市值数据，omega取老股中位数')
            omega_new = pd.Series(omega_old.median(), index=new_stocks, name='omega')
        else:
            omega_new = self._predict_new_stock_vol(
                params, intercept,
                new_in_ind.loc[common_new],
                new_in_mv.loc[common_new]
            )

        result = pd.concat([omega_old, omega_new]).sort_index()
        result.name = 'omega'
        return result

    def _compute_historical_vol(self, residuals: pd.DataFrame, as_of_date: str) -> pd.Series:
        """计算历史残差波动率（滚动窗口时间序列std）

        对每只股票取截止as_of_date的最近vol_rolling个交易日，
        计算时间序列维度上的标准差。

        Args:
            residuals: 残差收益率
            as_of_date: 截止日期

        Returns:
            Series(instrument -> omega)
        """
        as_of_ts = pd.Timestamp(as_of_date)
        dates = residuals.index.get_level_values('datetime')
        recent = residuals.loc[dates <= as_of_ts]

        col = recent.columns[0]
        n = self.vol_rolling

        def _tail_std(group: pd.DataFrame) -> float:
            """取最近vol_rolling天计算std"""
            return group[col].iloc[-n:].std()

        omega = recent.groupby(level='instrument').apply(_tail_std)
        omega.name = 'omega'
        return omega.dropna()

    def _classify_stock_age(
        self, residuals: pd.DataFrame, as_of_date: str
    ) -> tuple[list, list]:
        """按历史数据量分类老股票和新股

        Args:
            residuals: 残差收益率
            as_of_date: 截止日期

        Returns:
            (old_stocks, new_stocks)

        Raises:
            ValueError: 残差数据不足vol_window天
        """
        as_of_ts = pd.Timestamp(as_of_date)
        dates = residuals.index.get_level_values('datetime')
        mask = dates <= as_of_ts
        recent = residuals.loc[mask]

        n_days = recent.index.get_level_values('datetime').nunique()
        if n_days < self.vol_window:
            raise ValueError(
                f'残差数据不足: 仅{n_days}天，需要至少{self.vol_window}天。'
                f'请先运行 barra/risk_control 积累足够的残差数据'
            )

        # 统计每只股票的数据天数
        col = recent.columns[0]
        counts = recent.groupby(level='instrument')[col].count()

        old_stocks = counts[counts >= self.vol_window].index.tolist()
        new_stocks = counts[counts < self.vol_window].index.tolist()
        return old_stocks, new_stocks

    def _fit_regression(
        self,
        omega_old: pd.Series,
        industry_df: pd.DataFrame,
        market_cap_df: pd.DataFrame
    ) -> tuple[pd.Series, float]:
        """用老股票omega对行业+log(市值)做WLS回归

        Args:
            omega_old: 老股票omega, Series(instrument)
            industry_df: 行业数据, columns=['industry_code']
            market_cap_df: 市值数据, columns含'circ_mv'

        Returns:
            (params, intercept)
        """
        # 构建自变量矩阵
        X = self._build_regression_X(industry_df, market_cap_df)
        common = X.index.intersection(omega_old.index)
        X = X.loc[common]
        y = omega_old.loc[common]

        # 去除y中的NaN/inf
        valid = np.isfinite(y)
        X = X.loc[valid]
        y = y.loc[valid]

        params, intercept, _ = WLS(
            y.to_frame(), X, intercept=True, weight=1, verbose=True
        )
        logger.info(f'新股omega回归: {len(y)}个样本, {X.shape[1]}个自变量')
        return params, intercept

    def _predict_new_stock_vol(
        self,
        params: pd.Series,
        intercept: float,
        industry_df: pd.DataFrame,
        market_cap_df: pd.DataFrame
    ) -> pd.Series:
        """用回归模型为新股预测omega

        Args:
            params: 回归系数
            intercept: 截距
            industry_df: 新股行业数据
            market_cap_df: 新股市值数据

        Returns:
            Series(instrument -> omega_hat)
        """
        X = self._build_regression_X(industry_df, market_cap_df)
        # 对齐X的列与params的index
        missing_cols = params.index.difference(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[params.index]

        omega_hat = X.values @ params.values + intercept
        result = pd.Series(omega_hat, index=X.index, name='omega')
        # omega不应为负
        result = result.clip(lower=0.001)
        logger.info(f'新股omega预测: {len(result)}只')
        return result

    def _build_regression_X(
        self, industry_df: pd.DataFrame, market_cap_df: pd.DataFrame
    ) -> pd.DataFrame:
        """构建回归自变量矩阵：行业哑变量 + log(市值)

        Args:
            industry_df: columns=['industry_code']
            market_cap_df: columns含'circ_mv'

        Returns:
            DataFrame, index=instrument
        """
        common = industry_df.index.intersection(market_cap_df.index)
        ind = industry_df.loc[common, 'industry_code']
        mv = market_cap_df.loc[common, 'circ_mv']

        # 清洗：去除行业或市值为NaN的行
        valid = ind.notna() & mv.notna()
        ind = ind[valid]
        mv = mv[valid]

        # 行业哑变量
        dummies = pd.get_dummies(ind, prefix='ind', drop_first=True).astype(float)

        # log(流通市值)，万元单位
        log_mv = np.log(mv.astype(float).clip(lower=1)).rename('log_circ_mv')

        X = dummies.join(log_mv.to_frame(), how='inner')
        return X
