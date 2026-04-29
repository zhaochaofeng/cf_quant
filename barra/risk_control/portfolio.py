"""
组合管理模块 - 处理基准和持仓
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import random

from .data_loader import DataLoader


class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, market: str = 'csi300'):
        """
        初始化组合管理器
        
        Args:
            market: 市场代码
        """
        self.market = market
        self.data_loader = DataLoader(market=market)
    
    def get_benchmark_weights(self, calc_date: str) -> pd.Series:
        """
        获取基准权重（沪深300市值加权）
        
        Args:
            calc_date: 计算日期
            
        Returns:
            Series, index=instrument, values=weight
        """
        # 获取当日成分股
        instruments = self.data_loader.get_instruments(calc_date, calc_date)
        
        # 加载市值数据
        mv_df = self.data_loader.load_market_cap(instruments, calc_date, calc_date)
        
        # 直接计算当日总市值权重
        total_mv = mv_df['total_mv'].sum()
        if total_mv > 0:
            benchmark_weights = mv_df['total_mv'] / total_mv
            benchmark_weights.index = benchmark_weights.index.get_level_values('instrument')
            return benchmark_weights
        else:
            return pd.Series(dtype=float)
    
    def generate_random_portfolio(self, calc_date: str, n_stocks: int = 50, 
                                  random_state: Optional[int] = None) -> pd.Series:
        """
        生成随机投资组合（用于测试）
        
        Args:
            calc_date: 计算日期
            n_stocks: 随机选择的股票数量，默认50只
            random_state: 随机种子
            
        Returns:
            Series, index=instrument, values=weight（等权重）
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # 获取当日成分股
        instruments = self.data_loader.get_instruments(calc_date, calc_date)

        if len(instruments) == 0:
            raise Exception('No instruments found on the given date')

        if len(instruments) < n_stocks:
            n_stocks = len(instruments)
        
        # 随机选择n_stocks只股票
        selected = random.sample(instruments, n_stocks)
        
        # 等权重
        weights = pd.Series(1.0 / n_stocks, index=selected)
        
        return weights
    
    def load_portfolio(self, portfolio_input: Union[Dict[str, float], pd.Series, str],
                      calc_date: str) -> pd.Series:
        """
        加载投资组合权重
        
        Args:
            portfolio_input: 持仓输入，可以是：
                - dict: {instrument: weight}
                - pd.Series: index=instrument, values=weight
                - str: 'random'表示生成随机组合，或CSV文件路径
            calc_date: 计算日期
            
        Returns:
            Series, index=instrument, values=weight
        """
        if isinstance(portfolio_input, str):
            if portfolio_input.lower() == 'random':
                return self.generate_random_portfolio(calc_date)
            else:
                # 从CSV文件加载
                df = pd.read_csv(portfolio_input)
                weights = pd.Series(df['weight'].values, index=df['instrument'])
                return weights
        
        elif isinstance(portfolio_input, dict):
            weights = pd.Series(portfolio_input)
            # 归一化
            weights = weights / weights.sum()
            return weights
        
        elif isinstance(portfolio_input, pd.Series):
            weights = portfolio_input.copy()
            # 归一化
            weights = weights / weights.sum()
            return weights
        
        else:
            raise ValueError(f"不支持的持仓输入类型: {type(portfolio_input)}")
    
    @staticmethod
    def calculate_active_weights(portfolio_weights: pd.Series,
                                 benchmark_weights: pd.Series) -> pd.Series:
        """
        计算主动权重 h_PA = h_p - h_b
        
        Args:
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            
        Returns:
            Series, 主动权重
        """
        # 合并索引
        all_instruments = portfolio_weights.index.union(benchmark_weights.index)
        
        # 对齐权重
        h_p = portfolio_weights.reindex(all_instruments, fill_value=0.0)
        h_b = benchmark_weights.reindex(all_instruments, fill_value=0.0)
        
        # 计算主动权重
        h_pa = h_p - h_b
        
        return h_pa
    
    def get_portfolio_exposure(self, portfolio_weights: pd.Series, 
                               factor_exposure: pd.DataFrame) -> pd.Series:
        """
        计算组合因子暴露 x_p = X^T * h_p
        
        Args:
            portfolio_weights: 组合权重
            factor_exposure: 因子暴露矩阵，index=instrument, columns=factors
            
        Returns:
            Series, index=factors, values=exposure
        """
        # 对齐索引
        common_instruments = portfolio_weights.index.intersection(factor_exposure.index)
        h_p = portfolio_weights.reindex(common_instruments, fill_value=0.0)
        X = factor_exposure.loc[common_instruments]
        
        # 计算因子暴露
        x_p = X.T.dot(h_p)
        
        return x_p
    
    def save_portfolio(self, portfolio_weights: pd.Series, filepath: str):
        """
        保存投资组合到CSV文件
        
        Args:
            portfolio_weights: 组合权重
            filepath: 文件路径
        """
        df = pd.DataFrame({
            'instrument': portfolio_weights.index,
            'weight': portfolio_weights.values
        })
        df.to_csv(filepath, index=False, encoding='utf-8')
