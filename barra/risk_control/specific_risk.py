"""
特异风险矩阵估计模块 - ARMA + WLS 面板回归
"""
import pandas as pd
from typing import Tuple
from .config import INDUSTRY_NAMES
from utils import WLS, LoggerFactory, winsorize, TimeSeriesAnalysis

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

    def fit_arma(self, S: pd.Series) -> float:
        """
        对 S(t) 序列建立 ARMA 模型，预测 S(T+1)

        Args:
            S_series: S(t)序列, index=datetime

        Returns:
            S(T+1)预测值
        """
        logger.info(f'拟合 ARMA{self.arma_order} 模型...')
        try:
            # 去极值
            S = winsorize(S, method='median')
            # 标准化，防止数据尺度影响模型拟合
            mean = S.mean()
            std = S.std()
            S = (S - mean) / std
            tsa = TimeSeriesAnalysis(S)
            adf = tsa.adf(regression='c', autolag='BIC')
            pvalue = adf[1]
            if pvalue > 0.05:
                raise Exception(f'pvalue: {pvalue} > 0.05, ADF检验不通过，时间序列非平稳')
            order = tsa.order_select(max_ar=4, max_ma=2, ic="bic", trend="c")
            logger.info(f'ARMA 模型阶数: {order}')
            tsa.arma()
            logger.info(tsa.summary())
            pre = tsa.forecast(steps=1)
            S_T1 = pre.item() * std.item() + mean.item()
            self.S_forecast = S_T1
        except Exception as e:
            raise Exception(f'ARMA 拟合失败: {e}')
        return S_T1

    def estimate_specific_risk(
            self, residuals_df: pd.DataFrame,
            exposure_df: pd.DataFrame) -> pd.DataFrame:
        """
        估计特异风险矩阵

        流程：
        1. 对齐 residuals_df 和 exposure_df 索引
        2. 为 v_n 划分训练集（最近 panel_window 个交易日，排除最后1天）.
            v_n 训练区间： [-self.panel_window:-1]，预测日期：[-1]
            S_n 训练区间： [-self.panel_window:]，预测日期：未来一天（T+1）
        3. 分解训练集特异方差 → S_series, v_df
        4. ARMA 预测 S(T+1)
        5. 对 v_n 中位数去极值
        6. WLS 回归预测 v_n(T)并作为v_n(T+1) 的预测值
        7. 合成 u_n^2(T+1) = S(T+1) * [1 + v_n(T+1)]
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
        # 删除一个行业列
        exposure_df = exposure_df.drop(columns=f'ind_{INDUSTRY_NAMES[0]}')

        # 2. v_n 划分训练集和预测集
        dates = common_idx.get_level_values('datetime').unique().sort_values()

        # 使用最近 panel_window 个交易日作为 v_n 的计算区间
        if len(dates) > self.panel_window:
            v_dates = dates[-self.panel_window:]
        else:
            v_dates = dates
        v_train_dates = v_dates[:-1]
        v_latest_date = v_dates[-1]

        v_train_mask = common_idx.get_level_values('datetime').isin(v_train_dates)
        v_train_residuals = residuals_df.loc[v_train_mask]
        v_train_exposure = exposure_df.loc[v_train_mask].astype(float)
        v_latest_mask = common_idx.get_level_values('datetime') == v_latest_date
        v_latest_exposure = exposure_df.loc[v_latest_mask].astype(float)

        logger.info(f'v_n 训练期: {v_train_dates[0]} ~ {v_train_dates[-1]}, 共{len(v_train_dates)}期')
        logger.info(f'v_n 预测期: {v_latest_date}')

        # 3. 分解训练期特异方差
        S, _ = self.decompose_specific_variance(residuals_df)
        _, v_df = self.decompose_specific_variance(v_train_residuals)
        logger.info(f'S(t) 序列长度: {len(S)}')
        logger.info(f'v_n 序列长度: {len(v_df)}')

        # 4. ARMA 预测 S(T+1)
        S_T1 = self.fit_arma(S)

        # 5. 对 v_n 中位数去极值
        v_df = winsorize(v_df, method='median', level='datetime')

        # 6. WLS 回归 + 预测 v_n(t+1)
        merged = v_df.join(v_train_exposure, how='inner').dropna()
        y = merged[['v']]
        X = merged[v_train_exposure.columns].copy()

        slopes, intercept, _ = WLS(y, X, intercept=True)

        # 用最近日期因子暴露预测 v(T)
        v_T = v_latest_exposure @ slopes + intercept
        self.v_forecast = v_T

        logger.info(f'v(T) 预测完成，共{len(v_T)}只股票')

        # 7. 合成 u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        specific_var = (S_T1 * (1 + v_T)).clip(lower=1e-8)

        logger.info(f'特异方差范围: [{specific_var.min():.8f}, {specific_var.max():.8f}]')

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
