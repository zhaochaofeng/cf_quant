"""
特异风险矩阵估计模块 - ARMA + 面板回归
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional


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
        S_series = specific_var.groupby(level=1)['specific_var'].mean()
        
        # 计算 v_n(t) = u_n^2(t) / S(t) - 1
        v_list = []
        for date in specific_var.index.get_level_values(1).unique():
            date_var = specific_var.xs(date, level=1)['specific_var']
            S_t = S_series.loc[date]
            v_t = date_var / S_t - 1
            v_t.name = 'v'
            v_df = pd.DataFrame(v_t)
            v_df['date'] = date
            v_list.append(v_df)
        
        v_df = pd.concat(v_list)
        v_df = v_df.set_index('date', append=True)
        v_df = v_df.reorder_levels(['instrument', 'date'])
        
        self.S_series = S_series
        self.v_data = v_df
        
        print(f"分解完成，S(t)序列长度: {len(S_series)}")
        return S_series, v_df
    
    def fit_arma(self, S_series: pd.Series) -> float:
        """
        对S(t)序列建立ARMA模型并预测
        
        Args:
            S_series: S(t)序列
            
        Returns:
            S(t+1)预测值
        """
        print(f"拟合ARMA{self.arma_order}模型...")
        
        try:
            # 拟合ARMA模型
            model = ARIMA(S_series, order=(self.arma_order[0], 0, self.arma_order[1]))
            results = model.fit()
            
            # 预测下一期
            forecast = results.forecast(steps=1)
            S_forecast = forecast.iloc[0]
            
            print(f"ARMA模型拟合完成，AIC={results.aic:.2f}")
            print(f"S(t+1)预测值: {S_forecast:.6f}")
            
        except Exception as e:
            print(f"ARMA拟合失败: {str(e)}，使用历史均值代替")
            S_forecast = S_series.mean()
        
        self.S_forecast = S_forecast
        return S_forecast
    
    def panel_regression(self, v_df: pd.DataFrame, 
                        exposure_df: pd.DataFrame) -> pd.Series:
        """
        面板回归预测v_n(t+1)
        
        v_n(t) = sum_k(beta_k,n(t) * lambda_k(t)) + epsilon_n(t)
        
        使用混合回归（多期横截面数据合并）- 内存优化版本
        
        Args:
            v_df: v_n(t)数据，index=(instrument, date)
            exposure_df: 因子暴露数据，index=(instrument, date)
            
        Returns:
            v_n(t+1)预测值
        """
        print("进行面板回归...")
        
        # 获取所有日期
        dates = v_df.index.get_level_values(1).unique()
        print(f"   面板回归数据: {len(dates)} 个日期")
        
        # 分批处理：按日期分组进行回归
        batch_size = 10  # 每批处理10个日期
        all_y = []
        all_X = []
        
        for i in range(0, len(dates), batch_size):
            batch_dates = dates[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(dates) + batch_size - 1) // batch_size
            
            # 获取当前批次的v_df数据
            batch_v = v_df[v_df.index.get_level_values(1).isin(batch_dates)].copy()
            
            # 获取当前批次的exposure_df数据
            batch_exposure = exposure_df[exposure_df.index.get_level_values(1).isin(batch_dates)].copy()
            
            # 合并当前批次数据
            batch_merged = batch_v.join(batch_exposure, how='inner')
            batch_merged = batch_merged.dropna()
            
            if len(batch_merged) > 0:
                all_y.append(batch_merged['v'])
                all_X.append(batch_merged[exposure_df.columns])
            
            # 释放内存
            del batch_v, batch_exposure, batch_merged
            import gc
            gc.collect()
        
        if len(all_y) == 0:
            print("面板回归数据为空")
            return pd.Series(dtype=float)
        
        # 合并所有批次数据
        print(f"   合并 {len(all_y)} 批数据...")
        y = pd.concat(all_y)
        X = pd.concat(all_X)
        X = sm.add_constant(X)
        
        print(f"   回归数据形状: y={y.shape}, X={X.shape}")
        
        # 混合OLS回归
        try:
            model = sm.OLS(y, X).fit()
            print(f"面板回归完成，R²={model.rsquared:.4f}")
            
            # 获取回归系数
            lambda_coef = model.params.drop('const', errors='ignore')
            
        except Exception as e:
            print(f"面板回归失败: {str(e)}")
            return pd.Series(dtype=float)
        
        # 预测v_n(t+1) - 使用最新一期的因子暴露
        latest_date = exposure_df.index.get_level_values(1).max()
        latest_exposure = exposure_df.xs(latest_date, level=1)
        
        v_forecast = latest_exposure.dot(lambda_coef)
        
        # 去极值
        v_forecast = v_forecast.clip(lower=-0.5, upper=1.0)
        
        self.v_forecast = v_forecast
        print(f"v_n(t+1)预测完成，共{len(v_forecast)}只股票")
        
        return v_forecast
    
    def estimate_specific_risk(self, residuals_df: pd.DataFrame,
                               exposure_df: pd.DataFrame) -> pd.DataFrame:
        """
        估计特异风险矩阵
        
        完整流程：
        1. 分解特异方差
        2. ARMA预测S(t+1)
        3. 面板回归预测v_n(t+1)
        4. 合成u_n^2(t+1) = S(t+1) * [1 + v_n(t+1)]
        5. 构建对角矩阵Δ
        
        Args:
            residuals_df: 残差数据
            exposure_df: 因子暴露数据
            
        Returns:
            特异风险矩阵（对角矩阵）
        """
        print("=" * 60)
        print("开始估计特异风险矩阵...")
        
        # 1. 分解特异方差
        S_series, v_df = self.decompose_specific_variance(residuals_df)
        
        # 2. ARMA预测S(t+1)
        S_forecast = self.fit_arma(S_series)
        
        # 3. 面板回归预测v_n(t+1)
        v_forecast = self.panel_regression(v_df, exposure_df)
        
        # 4. 合成未来特异方差
        specific_var_forecast = S_forecast * (1 + v_forecast)
        
        # 确保非负
        specific_var_forecast = specific_var_forecast.clip(lower=1e-8)
        
        print(f"特异方差预测完成，范围: [{specific_var_forecast.min():.6f}, "
              f"{specific_var_forecast.max():.6f}]")
        
        # 5. 构建对角矩阵Δ（以DataFrame形式返回）
        delta_df = pd.DataFrame({
            'instrument': specific_var_forecast.index,
            'specific_var': specific_var_forecast.values
        })
        
        print("特异风险矩阵估计完成")
        print("=" * 60)
        
        return delta_df
    
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
