"""
特异风险矩阵估计模块 - ARMA + WLS 面板回归
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

from utils import WLS, LoggerFactory, winsorize

logger = LoggerFactory.get_logger(__name__)


class SpecificRiskEstimator:
    """特异风险矩阵估计器"""

    def __init__(self, arma_order: Tuple[int, int] = (1, 1),
                 panel_window: int = 120):
        """
        初始化特异风险估计器

        Args:
            arma_order: ARMA模型阶数(p, q)
            panel_window: 面板回归混合窗口（交易日）
        """
        self.arma_order = arma_order
        self.panel_window = panel_window
        self.S_forecast = None
        self.v_forecast = None

    def decompose_specific_variance(
            self, residuals_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        分解特异方差为 S(t) 和 v_n(t)

        S(t) = (1/N) * sum(u_n^2(t))
        v_n(t) = u_n^2(t) / S(t) - 1

        Args:
            residuals_df: 残差数据，index=(instrument, date), columns=['residual']

        Returns:
            (S_series, v_df)
        """
        logger.info('分解特异方差...')
        specific_var = residuals_df ** 2
        specific_var.columns = ['specific_var']

        S_series = specific_var.groupby(
            level='datetime')['specific_var'].mean()

        v_df = specific_var.groupby(
            level='datetime')['specific_var'].transform(
            lambda x: x / x.mean() - 1
        ).to_frame('v')

        logger.info(f'分解完成，S(t)序列长度: {len(S_series)}')
        return S_series, v_df

    def fit_arma(self, S_series: pd.Series) -> float:
        """
        对 S(t) 序列建立 ARMA 模型，预测 S(t+1)

        Args:
            S_series: S(t)序列, index=datetime

        Returns:
            S(t+1)预测值
        """
        logger.info(f'拟合 ARMA{self.arma_order} 模型...')
        try:
            model = ARIMA(
                S_series,
                order=(self.arma_order[0], 0, self.arma_order[1]),
            )
            results = model.fit()
            forecast = results.forecast(steps=1)
            S_t1 = forecast.iloc[0]
            logger.info(f'ARMA 拟合完成，AIC={results.aic:.2f}，'
                         f'S(t+1)={S_t1:.6f}')
        except Exception as e:
            logger.warning(f'ARMA 拟合失败: {e}，使用历史均值')
            S_t1 = S_series.mean()

        self.S_forecast = S_t1
        return S_t1

    def estimate_specific_risk(
            self, residuals_df: pd.DataFrame,
            exposure_df: pd.DataFrame) -> pd.DataFrame:
        """
        估计特异风险矩阵

        流程：
        1. 对齐 residuals_df 和 exposure_df 索引
        2. 划分训练集（最近 panel_window 个交易日，排除最后1天）
        3. 分解训练集特异方差 → S_series, v_df
        4. ARMA 预测 S(t+1)
        5. 对 v_n 中位数去极值
        6. WLS 回归预测 v_n(t+1)
        7. 合成 u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        8. 构建 N x N 对角矩阵 Δ

        Args:
            residuals_df: 残差数据, index=<instrument, datetime>
            exposure_df: 因子暴露数据, index=<instrument, datetime>

        Returns:
            pd.DataFrame: 特异方差，index=instrument, columns=['specific_var']
        """
        logger.info('=' * 60)
        logger.info('开始估计特异风险矩阵...')

        # 1. 按索引对齐
        common_idx = residuals_df.index.intersection(exposure_df.index)
        residuals_df = residuals_df.loc[common_idx]
        exposure_df = exposure_df.loc[common_idx]

        # 2. 划分训练集和预测集
        dates = common_idx.get_level_values('datetime').unique().sort_values()
        latest_date = dates[-1]

        # 使用最近 panel_window 个交易日（排除最后1天用于预测）
        if len(dates) > self.panel_window + 1:
            train_dates = dates[-(self.panel_window + 1):-1]
        else:
            train_dates = dates[:-1]

        train_mask = common_idx.get_level_values('datetime').isin(train_dates)

        train_residuals = residuals_df.loc[train_mask]
        train_exposure = exposure_df.loc[train_mask].astype(float)
        latest_mask = common_idx.get_level_values('datetime') == latest_date
        latest_exposure = exposure_df.loc[latest_mask].astype(float)

        logger.info(f'训练期: {train_dates[0]} ~ {train_dates[-1]}, '
                     f'共{len(train_dates)}期')
        logger.info(f'预测期: {latest_date}')

        # 3. 分解训练期特异方差
        S_series, v_df = self.decompose_specific_variance(train_residuals)

        # 4. ARMA 预测 S(t+1)
        S_t1 = self.fit_arma(S_series)

        # 5. 对 v_n 中位数去极值
        v_df = winsorize(v_df, method='median', level='datetime')

        # 6. WLS 回归 + 预测 v_n(t+1)
        merged = v_df.join(train_exposure, how='inner').dropna()
        y = merged[['v']]
        X = merged[train_exposure.columns].copy()

        slopes, intercept, _ = WLS(y, X, intercept=True)

        # 用最近日期因子暴露预测 v(t+1)
        v_t1 = latest_exposure @ slopes + intercept
        self.v_forecast = v_t1

        logger.info(f'v(t+1) 预测完成，共{len(v_t1)}只股票')

        # 7. 合成 u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        specific_var = (S_t1 * (1 + v_t1)).clip(lower=1e-8)

        logger.info(f'特异方差范围: [{specific_var.min():.6f}, '
                     f'{specific_var.max():.6f}]')

        # 8. 构建特异方差向量
        instruments = specific_var.index.get_level_values('instrument')
        delta = pd.DataFrame(
            {'specific_var': specific_var.values},
            index=instruments,
        )

        logger.info(f'特异方差向量: {delta.shape}')
        logger.info('特异风险估计完成')
        logger.info('=' * 60)

        return delta
