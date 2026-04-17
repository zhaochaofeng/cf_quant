"""
资产协方差矩阵计算模块
V = X * F * X^T + Delta
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class AssetCovarianceCalculator:
    """资产协方差矩阵计算器"""
    
    def __init__(self):
        """初始化计算器"""
        self.asset_covariance = None
        self.factor_covariance = None
        self.specific_risk = None
        self.exposure_matrix = None
    
    def calculate(self, exposure: pd.DataFrame,
                 factor_cov: pd.DataFrame,
                 specific_risk: pd.Series) -> pd.DataFrame:
        """
        计算资产协方差矩阵
        
        V = X * F * X^T + Delta
        
        Args:
            exposure: 因子暴露矩阵，index=instrument, columns=factors
            factor_cov: 因子协方差矩阵，index=factors, columns=factors
            specific_risk: 特异风险（方差），index=instrument
            
        Returns:
            资产协方差矩阵，index和columns都是instrument
        """
        logger.info('计算资产协方差矩阵...')
        
        # 对齐数据
        common_factors = exposure.columns.intersection(factor_cov.index)
        common_instruments = exposure.index.intersection(specific_risk.index)
        
        X = exposure.loc[common_instruments, common_factors].values
        F = factor_cov.loc[common_factors, common_factors].values
        delta_diag = specific_risk.loc[common_instruments].values
        
        # 计算 X * F * X^T
        try:
            XFXT = X @ F @ X.T
        except Exception as e:
            logger.error(f'矩阵乘法失败: {str(e)}')
            # 使用更稳定的方法
            XFXT = np.dot(np.dot(X, F), X.T)
        
        # 加上特异风险对角矩阵
        Delta = np.diag(delta_diag)
        V = XFXT + Delta
        
        # 转换为DataFrame
        V_df = pd.DataFrame(V, 
                           index=common_instruments,
                           columns=common_instruments)
        
        # 保存结果
        self.asset_covariance = V_df
        self.factor_covariance = factor_cov
        self.specific_risk = specific_risk
        self.exposure_matrix = exposure
        
        logger.info(f'资产协方差矩阵计算完成，维度: {V_df.shape}')
        return V_df
    
    def get_asset_covariance(self) -> pd.DataFrame:
        """
        获取资产协方差矩阵
        
        Returns:
            资产协方差矩阵
        """
        if self.asset_covariance is None:
            raise ValueError("资产协方差矩阵尚未计算")
        return self.asset_covariance
    
    def get_factor_component(self) -> pd.DataFrame:
        """
        获取因子风险成分（X * F * X^T）
        
        Returns:
            因子风险成分矩阵
        """
        if self.asset_covariance is None:
            raise ValueError("资产协方差矩阵尚未计算")
        
        X = self.exposure_matrix.values
        F = self.factor_covariance.values
        XFXT = X @ F @ X.T
        
        return pd.DataFrame(XFXT,
                           index=self.exposure_matrix.index,
                           columns=self.exposure_matrix.index)
    
    def get_specific_component(self) -> pd.DataFrame:
        """
        获取特异风险成分（对角矩阵Delta）
        
        Returns:
            特异风险对角矩阵
        """
        if self.asset_covariance is None:
            raise ValueError("资产协方差矩阵尚未计算")
        
        delta_diag = self.specific_risk.values
        Delta = np.diag(delta_diag)
        
        return pd.DataFrame(Delta,
                           index=self.specific_risk.index,
                           columns=self.specific_risk.index)
    
    def decompose_risk(self, portfolio_weights: pd.Series) -> dict:
        """
        分解组合风险
        
        Args:
            portfolio_weights: 组合权重
            
        Returns:
            风险分解结果
        """
        if self.asset_covariance is None:
            raise ValueError("资产协方差矩阵尚未计算")
        
        # 对齐权重
        common_instruments = portfolio_weights.index.intersection(self.asset_covariance.index)
        h = portfolio_weights.loc[common_instruments].values
        V = self.asset_covariance.loc[common_instruments, common_instruments].values
        
        # 组合总风险
        portfolio_variance = h.T @ V @ h
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # 因子风险成分
        X = self.exposure_matrix.loc[common_instruments].values
        F = self.factor_covariance.values
        factor_var = h.T @ X @ F @ X.T @ h
        
        # 特异风险成分
        delta = self.specific_risk.loc[common_instruments].values
        specific_var = np.sum((h ** 2) * delta)
        
        return {
            'portfolio_risk': portfolio_risk,
            'portfolio_variance': portfolio_variance,
            'factor_variance': factor_var,
            'specific_variance': specific_var,
            'factor_contribution_pct': factor_var / portfolio_variance * 100,
            'specific_contribution_pct': specific_var / portfolio_variance * 100,
        }
