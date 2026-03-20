"""
特异风险矩阵估计模块 - ARMA + 面板回归
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional

from utils import WLS


class SpecificRiskEstimator:
    """特异风险矩阵估计器"""
    
    def __init__(self, arma_order: Tuple[int, int] = (1, 1)):
        """
        初始化特异风险估计器
        
        Args:
            arma_order: ARMA模型阶数(p, q)
        """
        self.arma_order = arma_order
        self.S_series = {}  # 平均特异方差序列
        self.v_data = {}    # 相对偏离数据
        self.S_forecast = None
        self.v_forecast = None
    
    def decompose_specific_variance(self, residuals_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        分解特异方差为S(t)和v_n(t)
        
        S(t) = (1/N) * sum(u_n^2(t))
        v_n(t) = u_n^2(t) / S(t) - 1
        
        Args:
            residuals_df: 残差数据，index=(instrument, date), columns=['residual']
            
        Returns:
            (S_series, v_df)
        """
        print("分解特异方差...")
        
        # 计算特异方差 u_n^2(t)
        specific_var = residuals_df ** 2
        specific_var.columns = ['specific_var']
        
        # 计算 S(t) = 平均特异方差
        S_series = specific_var.groupby(level='datetime')['specific_var'].mean()
        
        # 计算 v_n(t) = u_n^2(t) / S(t) - 1
        v_df = specific_var.groupby(level='datetime')['specific_var'].transform(
            lambda x: x / x.mean() - 1
        ).to_frame('v')
        
        self.S_series = S_series
        self.v_data = v_df
        
        print(f"分解完成，S(t)序列长度: {len(S_series)}")
        return S_series, v_df
    
    def fit_arma(self, S_series: pd.Series) -> float:
        """
        对S(t)序列建立ARMA模型，预测S(t+1)
        
        Args:
            S_series: S(t)序列, index=datetime
            
        Returns:
            float: S(t+1)预测值
        """
        print(f"拟合ARMA{self.arma_order}模型...")
        
        try:
            model = ARIMA(S_series, order=(self.arma_order[0], 0, self.arma_order[1]))
            results = model.fit()
            forecast = results.forecast(steps=1)
            S_t1 = forecast.iloc[0]
            
            print(f"ARMA模型拟合完成，AIC={results.aic:.2f}")
            print(f"S(t+1)预测值: {S_t1:.6f}")
            
        except Exception as e:
            print(f"ARMA拟合失败: {str(e)}，使用历史均值")
            S_t1 = S_series.mean()
        
        self.S_forecast = S_t1
        return S_t1
    
    def panel_regression(self, v_df: pd.DataFrame, 
                        exposure_df: pd.DataFrame) -> pd.Series:
        """
        面板回归拟合v_n(t)
        
        使用混合回归，返回拟合值序列
        
        Args:
            v_df: v_n(t)数据，index=(instrument, date)
            exposure_df: 因子暴露数据（包含行业因子bool类型）
            
        Returns:
            v_n(t)拟合值序列，与v_df['v']相同shape和index
        """
        print("进行面板回归...")
        
        # 关键：将bool类型转换为数值类型
        exposure_df_numeric = exposure_df.astype(float)
        
        # 合并数据
        merged_df = v_df.join(exposure_df_numeric, how='inner').dropna()
        
        if len(merged_df) == 0:
            print("面板回归数据为空")
            return pd.Series(dtype=float)
        
        # 准备数据
        y = merged_df['v']
        X = merged_df[exposure_df_numeric.columns]
        X = sm.add_constant(X)
        
        print(f"   回归数据形状: y={y.shape}, X={X.shape}")
        
        # 混合OLS回归
        try:
            model = sm.OLS(y, X).fit()
            print(f"面板回归完成，R²={model.rsquared:.4f}")
            
            # 使用model.predict获取拟合值（方案B）
            v_fitted = model.predict(X)
            
        except Exception as e:
            print(f"面板回归失败: {str(e)}")
            # 异常处理：返回均值序列（保持index）
            v_fitted = pd.Series(y.mean(), index=y.index)
        
        print(f"面板回归拟合完成，共{len(v_fitted)}个观测值")
        return v_fitted
    
    def estimate_specific_risk(self, residuals_df: pd.DataFrame,
                               exposure_df: pd.DataFrame) -> pd.DataFrame:
        """
        估计特异风险矩阵
        
        完整流程：
        1. 对齐residuals_df和exposure_df索引
        2. 按时间划分训练集（历史）和预测集（最近日期）
        3. 分解训练集特异方差 -> S_series, v_df
        4. ARMA预测S(t+1)
        5. WLS回归预测v_n(t+1)
        6. 合成u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        7. 构建N×N对角矩阵Δ
        
        Args:
            residuals_df: 残差数据, index=<instrument, datetime>
            exposure_df: 因子暴露数据(包含行业因子), index=<instrument, datetime>
            
        Returns:
            pd.DataFrame: N×N对角矩阵，行列索引均为instrument
        """
        print('=' * 60)
        print('开始估计特异风险矩阵...')
        
        # 1. 按索引<instrument, datetime>对齐
        common_idx = residuals_df.index.intersection(exposure_df.index)
        residuals_df = residuals_df.loc[common_idx]
        exposure_df = exposure_df.loc[common_idx]
        
        # 2. 按时间划分：最近日期用于预测，其他日期用于训练
        dates = common_idx.get_level_values('datetime').unique().sort_values()
        latest_date = dates[-1]
        train_mask = common_idx.get_level_values('datetime') != latest_date
        
        train_residuals = residuals_df.loc[train_mask]
        train_exposure = exposure_df.loc[train_mask].astype(float)
        latest_exposure = exposure_df.loc[~train_mask].astype(float)
        
        print(f'训练期: {dates[0]} ~ {dates[-2]}, 共{len(dates)-1}期')
        print(f'预测期: {latest_date}')
        
        # 3. 分解训练期特异方差
        S_series, v_df = self.decompose_specific_variance(train_residuals)
        
        # 4. ARMA预测S(t+1)
        S_t1 = self.fit_arma(S_series)
        
        # 5. WLS回归 + 预测v_n(t+1)
        merged = v_df.join(train_exposure, how='inner').dropna()
        y = merged[['v']]
        X = merged[train_exposure.columns].copy()
        
        slopes, intercept, _ = WLS(y, X, intercept=True)
        
        # 用最近日期因子暴露预测v(t+1)
        v_t1 = latest_exposure @ slopes + intercept
        self.v_forecast = v_t1
        
        print(f'v(t+1)预测完成，共{len(v_t1)}只股票')
        
        # 6. 合成 u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        specific_var = (S_t1 * (1 + v_t1)).clip(lower=1e-8)
        
        print(f'特异方差预测完成，范围: [{specific_var.min():.6f}, '
              f'{specific_var.max():.6f}]')
        
        # 7. 构建N×N对角矩阵Δ，行列索引为instrument
        instruments = specific_var.index.get_level_values('instrument')
        delta = pd.DataFrame(
            np.diag(specific_var.values),
            index=instruments,
            columns=instruments
        )
        
        print(f'对角矩阵Δ形状: {delta.shape}')
        print('特异风险矩阵估计完成')
        print('=' * 60)
        
        return delta
    
    def get_specific_risk_matrix(self, instruments: list) -> np.ndarray:
        """
        获取特异风险对角矩阵
        
        Args:
            instruments: 股票列表
            
        Returns:
            对角矩阵
        """
        if self.v_forecast is None:
            raise ValueError("尚未估计特异风险")
        
        # 对齐股票
        specific_var = self.v_forecast.reindex(instruments, fill_value=0)
        specific_var = specific_var.clip(lower=1e-8)
        
        # 构建对角矩阵
        delta = np.diag(specific_var.values)
        
        return delta
