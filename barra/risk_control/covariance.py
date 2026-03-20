"""
协方差矩阵估计模块 - 因子协方差矩阵
"""
import pandas as pd
import numpy as np
from typing import Optional


class FactorCovarianceEstimator:
    """因子协方差矩阵估计器"""
    
    def __init__(self):
        """
        初始化协方差矩阵估计器
        """
        self.covariance_matrix = None
        self.mean_returns = None
    
    def estimate_sample_covariance(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        估计样本协方差矩阵
        
        F = 1/(T-1) * sum((b_t - b_bar) * (b_t - b_bar)^T)
        
        Args:
            factor_returns: 因子收益率数据，index=date, columns=factors
            min_periods: 最小观测期数
            
        Returns:
            协方差矩阵DataFrame，index和columns都是因子名称
        """
        print("估计因子协方差矩阵...")

        # 计算样本协方差矩阵。样本数 >= 因子数
        col_len = factor_returns.shape[1]
        cov_matrix = factor_returns.cov(min_periods=col_len)
        
        self.covariance_matrix = cov_matrix
        self.mean_returns = factor_returns.mean()
        
        print(f"协方差矩阵估计完成，使用{len(factor_returns)}期数据")
        print(f"矩阵维度: {cov_matrix.shape}")
        
        return cov_matrix
    
    def estimate_exponential_weighted_covariance(self, factor_returns: pd.DataFrame,
                                                 halflife: int = 36) -> pd.DataFrame:
        """
        指数加权协方差矩阵（EWMA）
        
        对近期数据赋予更高权重
        
        Args:
            factor_returns: 因子收益率数据
            halflife: 半衰期（月）
            
        Returns:
            协方差矩阵DataFrame
        """
        print(f"估计指数加权协方差矩阵（半衰期={halflife}个月）...")
        
        T = len(factor_returns)
        
        # 计算指数权重
        decay = 0.5 ** (1 / halflife)
        weights = np.array([decay ** (T - 1 - i) for i in range(T)])
        weights = weights / weights.sum()
        
        # 计算加权均值
        weighted_mean = (factor_returns.T * weights).T.sum()
        
        # 计算加权协方差
        centered = factor_returns - weighted_mean
        cov_matrix = pd.DataFrame(
            np.dot(centered.T * weights, centered),
            index=factor_returns.columns,
            columns=factor_returns.columns
        )
        
        self.covariance_matrix = cov_matrix
        self.mean_returns = weighted_mean
        
        print(f"指数加权协方差矩阵估计完成")
        return cov_matrix
    
    def shrinkage_covariance(self, factor_returns: pd.DataFrame,
                            shrinkage_target: str = 'constant_correlation',
                            delta: Optional[float] = None) -> pd.DataFrame:
        """
        Ledoit-Wolf收缩估计
        
        将样本协方差向目标矩阵收缩，提高稳定性
        
        Args:
            factor_returns: 因子收益率数据
            shrinkage_target: 收缩目标类型
            delta: 收缩强度（0-1），None则自动估计
            
        Returns:
            收缩后的协方差矩阵
        """
        print("执行收缩估计...")
        
        # 计算样本协方差
        sample_cov = factor_returns.cov()
        
        if shrinkage_target == 'constant_correlation':
            # 常数相关矩阵目标
            variances = np.diag(sample_cov)
            std = np.sqrt(variances)
            
            # 计算平均相关系数
            corr_matrix = np.corrcoef(factor_returns.T.dropna())
            np.fill_diagonal(corr_matrix, 0)
            avg_corr = corr_matrix.mean()
            
            # 构建目标矩阵
            target = np.outer(std, std) * avg_corr
            np.fill_diagonal(target, variances)
            target = pd.DataFrame(target, 
                                 index=sample_cov.index, 
                                 columns=sample_cov.columns)
        else:
            target = pd.DataFrame(np.diag(np.diag(sample_cov)),
                                 index=sample_cov.index,
                                 columns=sample_cov.columns)
        
        # 自动估计收缩强度（简化版）
        if delta is None:
            # 使用简单启发式：数据越少，收缩越强
            T = len(factor_returns)
            K = len(factor_returns.columns)
            delta = K / T
            delta = min(delta, 0.5)  # 上限0.5
        
        # 收缩估计
        shrunk_cov = delta * target + (1 - delta) * sample_cov
        
        self.covariance_matrix = shrunk_cov
        print(f"收缩估计完成，收缩强度={delta:.4f}")
        
        return shrunk_cov
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        获取协方差矩阵
        
        Returns:
            协方差矩阵
        """
        if self.covariance_matrix is None:
            raise ValueError("协方差矩阵尚未估计")
        return self.covariance_matrix
    
    def get_factor_volatility(self) -> pd.Series:
        """
        获取因子波动率（协方差矩阵对角线的平方根）
        
        Returns:
            因子波动率序列
        """
        if self.covariance_matrix is None:
            raise ValueError("协方差矩阵尚未估计")
        
        vol = np.sqrt(np.diag(self.covariance_matrix))
        return pd.Series(vol, index=self.covariance_matrix.index)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        获取相关系数矩阵
        
        Returns:
            相关系数矩阵
        """
        if self.covariance_matrix is None:
            raise ValueError("协方差矩阵尚未估计")
        
        cov = self.covariance_matrix.values
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        
        return pd.DataFrame(corr, 
                           index=self.covariance_matrix.index,
                           columns=self.covariance_matrix.columns)
    
    def save(self, filepath: str):
        """
        保存协方差矩阵到文件
        
        Args:
            filepath: 文件路径
        """
        if self.covariance_matrix is not None:
            self.covariance_matrix.to_csv(filepath)
            print(f"协方差矩阵已保存至: {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载协方差矩阵
        
        Args:
            filepath: 文件路径
        """
        self.covariance_matrix = pd.read_csv(filepath, index_col=0)
        print(f"协方差矩阵已从{filepath}加载")
