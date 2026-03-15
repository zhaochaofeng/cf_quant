"""
风险归因分析模块 - MCAR/RCAR/FMCAR/FRCAR计算
"""
import pandas as pd
import numpy as np
from typing import Tuple


class RiskAttributionAnalyzer:
    """风险归因分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.mcar = None
        self.rcar = None
        self.fmcar = None
        self.frcar = None
    
    def calculate_mcar(self, asset_cov: pd.DataFrame,
                      active_weights: pd.Series,
                      active_risk: float) -> pd.Series:
        """
        计算股票的主动风险边际贡献（MCAR）
        
        MCAR = (V * h_PA) / psi_p
        
        Args:
            asset_cov: 资产协方差矩阵
            active_weights: 主动权重
            active_risk: 主动风险（跟踪误差）
            
        Returns:
            MCAR序列
        """
        # 对齐数据
        common_idx = active_weights.index.intersection(asset_cov.index)
        h_pa = active_weights.loc[common_idx].values
        V = asset_cov.loc[common_idx, common_idx].values
        
        # 计算 V * h_PA
        Vh = V @ h_pa
        
        # 计算 MCAR - 防止除零
        if active_risk < 1e-10:
            print(f"警告：主动风险接近零 ({active_risk:.10f})，MCAR设为0")
            mcar = np.zeros_like(Vh)
        else:
            mcar = Vh / active_risk
        
        mcar_series = pd.Series(mcar, index=common_idx)
        self.mcar = mcar_series
        
        return mcar_series
    
    def calculate_rcar(self, active_weights: pd.Series,
                      mcar: pd.Series) -> pd.Series:
        """
        计算股票的主动风险贡献（RCAR）
        
        RCAR = h_PA ⊙ MCAR
        
        Args:
            active_weights: 主动权重
            mcar: MCAR序列
            
        Returns:
            RCAR序列
        """
        # 对齐
        common_idx = active_weights.index.intersection(mcar.index)
        h_pa = active_weights.loc[common_idx]
        mcar_aligned = mcar.loc[common_idx]
        
        # 逐元素相乘
        rcar = h_pa * mcar_aligned
        
        self.rcar = rcar
        return rcar
    
    def calculate_fmcar(self, factor_cov: pd.DataFrame,
                       exposure: pd.DataFrame,
                       active_weights: pd.Series,
                       active_risk: float) -> pd.Series:
        """
        计算因子的主动风险边际贡献（FMCAR）
        
        FMCAR = (F * x_PA) / psi_p
        其中 x_PA = X^T * h_PA
        
        Args:
            factor_cov: 因子协方差矩阵
            exposure: 因子暴露矩阵
            active_weights: 主动权重
            active_risk: 主动风险
            
        Returns:
            FMCAR序列
        """
        # 计算主动因子暴露 x_PA = X^T * h_PA
        common_idx = active_weights.index.intersection(exposure.index)
        h_pa = active_weights.loc[common_idx]
        X = exposure.loc[common_idx]
        
        x_pa = X.T @ h_pa
        
        # 对齐因子
        common_factors = x_pa.index.intersection(factor_cov.index)
        x_pa = x_pa.loc[common_factors].values
        F = factor_cov.loc[common_factors, common_factors].values
        
        # 计算 F * x_PA
        Fx = F @ x_pa
        
        # 计算 FMCAR - 防止除零
        if active_risk < 1e-10:
            print(f"警告：主动风险接近零 ({active_risk:.10f})，FMCAR设为0")
            fmcar = np.zeros_like(Fx)
        else:
            fmcar = Fx / active_risk
        
        fmcar_series = pd.Series(fmcar, index=common_factors)
        self.fmcar = fmcar_series
        
        return fmcar_series
    
    def calculate_frcar(self, exposure: pd.DataFrame,
                       active_weights: pd.Series,
                       fmcar: pd.Series) -> pd.Series:
        """
        计算因子的主动风险贡献（FRCAR）
        
        FRCAR = x_PA ⊙ FMCAR
        
        Args:
            exposure: 因子暴露矩阵
            active_weights: 主动权重
            fmcar: FMCAR序列
            
        Returns:
            FRCAR序列
        """
        # 计算主动因子暴露
        common_idx = active_weights.index.intersection(exposure.index)
        h_pa = active_weights.loc[common_idx]
        X = exposure.loc[common_idx]
        
        x_pa = X.T @ h_pa
        
        # 对齐
        common_factors = x_pa.index.intersection(fmcar.index)
        x_pa_aligned = x_pa.loc[common_factors]
        fmcar_aligned = fmcar.loc[common_factors]
        
        # 逐元素相乘
        frcar = x_pa_aligned * fmcar_aligned
        
        self.frcar = frcar
        return frcar
    
    def calculate_active_risk(self, asset_cov: pd.DataFrame,
                             active_weights: pd.Series) -> float:
        """
        计算主动风险（跟踪误差）
        
        psi_p = sqrt(h_PA^T * V * h_PA)
        
        Args:
            asset_cov: 资产协方差矩阵
            active_weights: 主动权重
            
        Returns:
            主动风险值
        """
        # 对齐
        common_idx = active_weights.index.intersection(asset_cov.index)
        h_pa = active_weights.loc[common_idx].values
        V = asset_cov.loc[common_idx, common_idx].values
        
        # 计算
        variance = h_pa.T @ V @ h_pa
        active_risk = np.sqrt(variance)
        
        return active_risk
    
    def calculate_total_risk(self, asset_cov: pd.DataFrame,
                            portfolio_weights: pd.Series) -> float:
        """
        计算组合总风险
        
        sigma_p = sqrt(h_p^T * V * h_p)
        
        Args:
            asset_cov: 资产协方差矩阵
            portfolio_weights: 组合权重
            
        Returns:
            组合总风险
        """
        common_idx = portfolio_weights.index.intersection(asset_cov.index)
        h_p = portfolio_weights.loc[common_idx].values
        V = asset_cov.loc[common_idx, common_idx].values
        
        variance = h_p.T @ V @ h_p
        total_risk = np.sqrt(variance)
        
        return total_risk
    
    def analyze_risk(self, asset_cov: pd.DataFrame,
                    factor_cov: pd.DataFrame,
                    exposure: pd.DataFrame,
                    portfolio_weights: pd.Series,
                    benchmark_weights: pd.Series) -> dict:
        """
        执行完整的风险归因分析
        
        Args:
            asset_cov: 资产协方差矩阵
            factor_cov: 因子协方差矩阵
            exposure: 因子暴露矩阵
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            
        Returns:
            完整的风险分析结果
        """
        print("=" * 60)
        print("开始风险归因分析...")
        
        # 计算主动权重
        all_instruments = portfolio_weights.index.union(benchmark_weights.index)
        h_p = portfolio_weights.reindex(all_instruments, fill_value=0.0)
        h_b = benchmark_weights.reindex(all_instruments, fill_value=0.0)
        h_pa = h_p - h_b
        
        # 计算风险指标
        total_risk = self.calculate_total_risk(asset_cov, h_p)
        active_risk = self.calculate_active_risk(asset_cov, h_pa)
        
        print(f"组合总风险: {total_risk:.6f}")
        print(f"主动风险(跟踪误差): {active_risk:.6f}")
        
        # 计算MCAR和RCAR
        mcar = self.calculate_mcar(asset_cov, h_pa, active_risk)
        rcar = self.calculate_rcar(h_pa, mcar)
        
        # 验证：RCAR之和应等于主动风险
        rcar_sum = rcar.sum()
        print(f"RCAR之和: {rcar_sum:.6f} (应等于主动风险 {active_risk:.6f})")
        
        # 计算FMCAR和FRCAR
        fmcar = self.calculate_fmcar(factor_cov, exposure, h_pa, active_risk)
        frcar = self.calculate_frcar(exposure, h_pa, fmcar)
        
        # 验证：FRCAR之和应约等于主动风险减去特异风险贡献
        frcar_sum = frcar.sum()
        print(f"FRCAR之和: {frcar_sum:.6f}")
        
        print("风险归因分析完成")
        print("=" * 60)
        
        return {
            'total_risk': total_risk,
            'active_risk': active_risk,
            'mcar': mcar,
            'rcar': rcar,
            'fmcar': fmcar,
            'frcar': frcar,
            'rcar_sum': rcar_sum,
            'frcar_sum': frcar_sum,
        }
