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
    
    def fit_arma(self, S_series: pd.Series) -> pd.Series:
        """
        对S(t)序列建立ARMA模型并返回拟合值
        
        拟合ARMA模型，并返回与输入序列相同长度的拟合值序列
        
        Args:
            S_series: S(t)序列
            
        Returns:
            pd.Series: S(t)拟合值序列，与输入S_series相同shape和index
        """
        print(f"拟合ARMA{self.arma_order}模型...")
        
        try:
            # 拟合ARMA模型
            model = ARIMA(S_series, order=(self.arma_order[0], 0, self.arma_order[1]))
            results = model.fit()
            
            # 获取拟合值（与S_series相同长度和index）
            fitted_values = results.fittedvalues
            
            print(f"ARMA模型拟合完成，AIC={results.aic:.2f}")
            print(f"拟合值序列长度: {len(fitted_values)}")
            
        except Exception as e:
            print(f"ARMA拟合失败: {str(e)}，使用历史均值序列")
            # 如果拟合失败，返回均值序列（保持相同shape和index）
            fitted_values = pd.Series(S_series.mean(), index=S_series.index)
        
        self.S_forecast = fitted_values
        return fitted_values
    
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
        1. 分解特异方差
        2. ARMA预测S(t)
        3. 面板回归预测v_n(t)
        4. 合成u_n^2(t) = S(t) * [1 + v_n(t)]
        5. 构建对角矩阵Δ
        
        Args:
            residuals_df: 残差数据
            exposure_df: 因子暴露数据(包含行业因子)
            
        Returns:
            特异风险矩阵（对角矩阵）
        """
        print("=" * 60)
        print("开始估计特异风险矩阵...")
        
        # 1. 分解特异方差
        S_series, v_df = self.decompose_specific_variance(residuals_df)
        
        # 2. ARMA预测S(t)
        S_forecast = self.fit_arma(S_series)
        
        # 3. 面板回归预测v_n(t)
        v_forecast = self.panel_regression(v_df, exposure_df)
        
        # 4. 对齐日期索引并合成未来特异方差
        # S_forecast: index=date, v_forecast: index=(instrument, date)
        # 将S_forecast按照日期对齐到v_forecast的每个股票
        v_dates = v_forecast.index.get_level_values(1)
        S_aligned = S_forecast.reindex(v_dates)
        
        # 逐元素相乘: u_n^2(t) = S(t) * [1 + v_n(t)]
        specific_var_values = S_aligned.values * (1 + v_forecast.values)
        specific_var_forecast = pd.Series(specific_var_values, index=v_forecast.index)
        
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
