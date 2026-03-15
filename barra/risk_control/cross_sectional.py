"""
横截面回归模块 - 加权最小二乘估计因子收益率
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Optional


class CrossSectionalRegression:
    """横截面回归估计器"""
    
    def __init__(self, weight_type: str = 'sqrt_market_cap'):
        """
        初始化横截面回归器
        
        Args:
            weight_type: 权重类型，默认'市值平方根'
        """
        self.weight_type = weight_type
        self.factor_returns = {}  # 存储各期因子收益率
        self.residuals = {}       # 存储各期残差
    
    def calculate_weights(self, market_cap: pd.Series) -> pd.Series:
        """
        计算回归权重（市值平方根）
        
        Args:
            market_cap: 市值序列
            
        Returns:
            权重序列
        """
        # 市值平方根加权
        weights = np.sqrt(market_cap)
        # 归一化（可选）
        # weights = weights / weights.sum()
        return weights
    
    def fit(self, date: str, returns: pd.Series, 
            exposure: pd.DataFrame, market_cap: pd.Series) -> dict:
        """
        单期横截面回归
        
        模型：r_t = X_t * b_t + u_t
        
        Args:
            date: 日期
            returns: 股票收益率序列，index=instrument
            exposure: 因子暴露矩阵，index=instrument, columns=factors
            market_cap: 市值序列，index=instrument
            
        Returns:
            回归结果字典，包含factor_returns, residuals, r_squared等
        """
        # 对齐数据
        common_index = returns.index.intersection(exposure.index).intersection(market_cap.index)
        r = returns.loc[common_index]
        X = exposure.loc[common_index]
        mv = market_cap.loc[common_index]
        
        # 强制转换为数值类型（关键修复）
        r = pd.to_numeric(r, errors='coerce')
        X = X.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        mv = pd.to_numeric(mv, errors='coerce')
        
        # 处理缺失值
        valid_mask = r.notna() & X.notna().all(axis=1) & mv.notna()
        r = r[valid_mask]
        X = X[valid_mask]
        mv = mv[valid_mask]
        
        if len(r) == 0 or X.shape[1] == 0:
            print(f"{date}: 无有效数据，跳过回归")
            return None
        
        # 再次确保数据类型为float
        r = r.astype(float)
        X = X.astype(float)
        mv = mv.astype(float)
        
        # 计算权重
        weights = self.calculate_weights(mv)
        
        # 加权最小二乘回归
        # 注意：statsmodels的WLS需要权重矩阵的对角线元素
        model = sm.WLS(r, X, weights=weights)
        
        try:
            results = model.fit()
        except Exception as e:
            print(f"{date}: 回归失败 - {str(e)}")
            return None
        
        # 提取结果
        factor_returns = results.params
        residuals = results.resid
        
        # 保存结果
        self.factor_returns[date] = factor_returns
        self.residuals[date] = residuals
        
        return {
            'date': date,
            'factor_returns': factor_returns,
            'residuals': residuals,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'n_obs': results.nobs,
            'f_statistic': results.fvalue,
            'f_pvalue': results.f_pvalue,
        }
    
    def fit_multi_periods(self, returns_df: pd.DataFrame,
                         exposure_df: pd.DataFrame,
                         market_cap_df: pd.DataFrame,
                         freq: str = 'month') -> pd.DataFrame:
        """
        多期横截面回归
        
        在Barra模型中，横截面回归应该按月进行，只在每月最后一个交易日回归一次。
        
        Args:
            returns_df: 收益率数据，index=(instrument, date)
            exposure_df: 因子暴露数据，index=(instrument, date)
            market_cap_df: 市值数据，index=(instrument, date)
            freq: 回归频率，'month'为月频（默认），'day'为日频
            
        Returns:
            因子收益率矩阵，index=date, columns=factors
        """
        print(f"开始多期横截面回归（频率: {freq}）...")
        
        # 获取所有日期
        dates = returns_df.index.get_level_values(1).unique()
        
        if freq == 'month':
            # 按月频：只取每月最后一个交易日
            dates_df = pd.DataFrame({'date': dates})
            dates_df['year_month'] = dates_df['date'].dt.to_period('M')
            # 取每月最后一天
            monthly_dates = dates_df.groupby('year_month')['date'].max()
            regression_dates = monthly_dates.tolist()
            print(f"   月频模式：从 {len(dates)} 个交易日中选取 {len(regression_dates)} 个月末日期")
        else:
            # 按日频：所有交易日（不建议，仅用于测试）
            regression_dates = dates.tolist()
            print(f"   日频模式：共 {len(regression_dates)} 个交易日")
        
        results_list = []
        for date in regression_dates:
            # 提取当期数据
            r = returns_df.xs(date, level=1).iloc[:, 0]
            X = exposure_df.xs(date, level=1)
            mv = market_cap_df.xs(date, level=1).iloc[:, 0]
            
            result = self.fit(str(date), r, X, mv)
            if result:
                results_list.append(result)
        
        # 构建因子收益率DataFrame
        if results_list:
            factor_returns_df = pd.DataFrame(
                {r['date']: r['factor_returns'] for r in results_list}
            ).T
            factor_returns_df.index.name = 'date'
        else:
            factor_returns_df = pd.DataFrame()
        
        print(f"横截面回归完成，共{len(results_list)}期")
        return factor_returns_df
    
    def get_factor_returns(self) -> pd.DataFrame:
        """
        获取所有期的因子收益率
        
        Returns:
            DataFrame, index=date, columns=factors
        """
        if not self.factor_returns:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.factor_returns).T
        df.index.name = 'date'
        return df
    
    def get_residuals(self) -> pd.DataFrame:
        """
        获取所有期的残差
        
        Returns:
            DataFrame, index=(instrument, date), columns=['residual']
        """
        if not self.residuals:
            return pd.DataFrame()
        
        # 构建DataFrame
        resid_list = []
        for date, resid in self.residuals.items():
            temp_df = pd.DataFrame({
                'instrument': resid.index,
                'date': date,
                'residual': resid.values
            })
            resid_list.append(temp_df)
        
        df = pd.concat(resid_list, ignore_index=True)
        df = df.set_index(['instrument', 'date'])
        return df
    
    def calculate_residual_variance(self, date: str) -> pd.Series:
        """
        计算特异方差
        
        Args:
            date: 日期
            
        Returns:
            特异方差序列
        """
        if date not in self.residuals:
            return pd.Series(dtype=float)
        
        residuals = self.residuals[date]
        # 特异方差 = 残差平方
        specific_var = residuals ** 2
        
        return specific_var
