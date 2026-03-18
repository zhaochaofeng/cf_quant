"""
因子暴露矩阵构建模块
包含：原始因子计算、去极值、中性化、正交化、标准化、行业因子合并
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .output import RiskOutputManager

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.multiprocess import multiprocessing_wrapper

from .config import STYLE_FACTOR_LIST, FACTOR_FUNCTIONS, INDUSTRY_MAPPING


class FactorExposureBuilder:
    """因子暴露矩阵构建器"""

    def calculate_raw_factors(self, raw_data: pd.DataFrame,
                              n_jobs: int = 1
                              ) -> pd.DataFrame:
        """
        计算原始CNE6因子值
        
        Args:
            raw_data: 原始数据DataFrame，包含所有必要字段
            n_jobs: 并行进程数
            
        Returns:
            DataFrame, index=(instrument, datetime), columns=因子名称
        """
        print("开始计算原始因子值...")
        raw_data = raw_data.sort_index()
        
        # 准备并行计算任务
        factor_results = {}
        
        if n_jobs > 1:
            # 并行计算
            func_calls = []
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    func_calls.append((
                        self._compute_single_factor,
                        (raw_data, factor_name, FACTOR_FUNCTIONS[factor_name])
                    ))
            
            results = multiprocessing_wrapper(func_calls, n=n_jobs)
            for factor_name, result in results:
                if result is not None:
                    factor_results[factor_name] = result
        else:
            # 串行计算
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    _, result = self._compute_single_factor(
                        raw_data, factor_name, FACTOR_FUNCTIONS[factor_name]
                    )
                    if result is not None:
                        factor_results[factor_name] = result
        
        # 合并所有因子
        factor_df = pd.DataFrame(index=raw_data.index)
        for factor_name, series in factor_results.items():
            factor_df[factor_name] = series
        
        print(f"原始因子计算完成，共{len(factor_df.columns)}个因子")

        return factor_df
    
    def _compute_single_factor(self, raw_data: pd.DataFrame, 
                               factor_name: str, 
                               factor_func) -> tuple:
        """
        计算单个因子（用于并行）
        raw_data：原始字段数据
        factor_name：因子名称
        factor_func：因子计算函数
        
        Returns:
            (factor_name, series)
        """
        try:
            result = factor_func(raw_data)
            if result is not None and not result.empty:
                series = result.iloc[:, 0]  # 取第一列
                return factor_name, series
        except Exception as e:
            err_msg = f"因子{factor_name}计算失败: {str(e)}"
            # raise Exception(err_msg)
            print(err_msg)
            return factor_name, None
    
    def winsorize_factors(self, factor_df: pd.DataFrame, 
                         method: str = 'median') -> pd.DataFrame:
        """
        因子去极值处理（中位数去极值）
        
        Args:
            factor_df: 因子数据
            method: 去极值方法，'median'或'quantile'
            
        Returns:
            去极值后的因子数据
        """
        print("进行中位数去极值...")
        result_df = pd.DataFrame(index=factor_df.index)
        
        # 按日期分组处理
        for date, group in factor_df.groupby(level=1):
            for factor in factor_df.columns:
                values = group[factor].dropna()
                if len(values) == 0:
                    continue
                
                if method == 'median':
                    # 中位数去极值
                    median = values.median()
                    mad = np.median(np.abs(values - median))
                    lower_bound = median - 5 * 1.4826 * mad
                    upper_bound = median + 5 * 1.4826 * mad
                else:
                    # 分位数去极值
                    lower_bound = values.quantile(0.01)
                    upper_bound = values.quantile(0.99)
                
                # 截断
                clipped = values.clip(lower=lower_bound, upper=upper_bound)
                result_df.loc[group.index, factor] = clipped
        
        return result_df
    
    def neutralize_factors(self, factor_df: pd.DataFrame, 
                          industry_df: pd.DataFrame,
                          market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """
        行业/市值中性化
        
        参考jqfactor_analyzer的实现逻辑：
        对每个因子，用行业和市值做回归，取残差
        
        Args:
            factor_df: 因子数据
            industry_df: 行业数据，index=(instrument, datetime), 值为行业代码
            market_cap_df: 市值数据，index=(instrument, datetime), columns包含'circ_mv'
            
        Returns:
            中性化后的因子数据
        """
        print("进行行业/市值中性化...")
        result_df = pd.DataFrame(index=factor_df.index)
        
        # 准备行业虚拟变量
        industry_dummies = pd.get_dummies(industry_df.iloc[:, 0], prefix='ind')
        
        # 合并数据
        merged_df = factor_df.join(industry_dummies, how='inner')
        merged_df = merged_df.join(market_cap_df[['circ_mv']], how='inner')
        
        # 对每个因子进行中性化
        for factor in factor_df.columns:
            neutralized = self._neutralize_single_factor(
                merged_df, factor, industry_dummies.columns.tolist()
            )
            result_df[factor] = neutralized
        
        return result_df
    
    def _neutralize_single_factor(self, merged_df: pd.DataFrame,
                                  factor_name: str,
                                  industry_cols: List[str]) -> pd.Series:
        """
        对单个因子进行中性化

        Returns:
            中性化后的因子序列
        """
        y = merged_df[factor_name]
        # 确保 y 是数值类型
        y = pd.to_numeric(y, errors='coerce')

        # 自变量：行业虚拟变量 + 对数市值
        X_cols = industry_cols + ['circ_mv']
        X = merged_df[X_cols].copy()

        # 确保所有列都是数值类型
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # 计算对数市值，处理可能的inf和负数
        X['circ_mv'] = np.log(X['circ_mv'].clip(lower=1e-10))
        X = sm.add_constant(X)

        # 加权最小二乘（市值平方根加权）
        weights = np.sqrt(pd.to_numeric(merged_df['circ_mv'], errors='coerce').clip(lower=1e-10))

        # 移除包含NaN的行
        valid_idx = y.notna() & X.notna().all(axis=1) & weights.notna()
        if valid_idx.sum() < len(X.columns) + 1:  # 至少需要比变量数多的样本
            print(f"警告：因子{factor_name}有效样本不足，跳过中性化")
            return pd.Series(np.nan, index=merged_df.index)

        y_clean = y[valid_idx]
        X_clean = X[valid_idx]
        weights_clean = weights[valid_idx]

        # 再次确保数据类型正确
        y_clean = y_clean.astype(float)
        X_clean = X_clean.astype(float)
        weights_clean = weights_clean.astype(float)

        try:
            # 拟合模型
            model = sm.WLS(y_clean, X_clean, weights=weights_clean)
            results = model.fit()

            # 返回残差
            resid = pd.Series(np.nan, index=merged_df.index, dtype=float)
            resid.loc[valid_idx] = results.resid.values
        except Exception as e:
            print(f"警告：因子{factor_name}中性化失败: {str(e)}")
            resid = pd.Series(np.nan, index=merged_df.index, dtype=float)

        return resid
    
    def orthogonalize_factors(self, factor_df: pd.DataFrame,
                             factor_order: Optional[List[str]] = None) -> pd.DataFrame:
        """
        因子正交化
        
        按照指定顺序，从第2个因子开始，以当前因子为因变量，
        排在前面的因子为自变量，进行多元线性回归拟合，取回归残差
        
        Args:
            factor_df: 因子数据
            factor_order: 因子顺序列表，默认使用STYLE_FACTOR_LIST
            
        Returns:
            正交化后的因子数据
        """
        print("进行因子正交化...")
        if factor_order is None:
            factor_order = [f for f in STYLE_FACTOR_LIST if f in factor_df.columns]
        
        result_df = pd.DataFrame(index=factor_df.index)
        
        for i, factor_name in enumerate(factor_order):
            if factor_name not in factor_df.columns:
                continue
            
            if i == 0:
                # 第一个因子保持不变
                result_df[factor_name] = factor_df[factor_name]
            else:
                # 对前面的因子进行回归
                y = factor_df[factor_name]
                X = result_df.iloc[:, :i]  # 前面已正交化的因子
                X = sm.add_constant(X)
                
                # 回归并取残差
                model = sm.OLS(y, X, missing='drop').fit()
                resid = pd.Series(index=factor_df.index, dtype=float)
                resid.loc[model.resid.index] = model.resid.values
                result_df[factor_name] = resid
        
        return result_df
    
    def standardize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        因子标准化（Z-Score）
        
        Args:
            factor_df: 因子数据
            
        Returns:
            标准化后的因子数据（均值0，标准差1）
        """
        print("进行标准化...")
        result_df = pd.DataFrame(index=factor_df.index)
        
        # 按日期分组标准化
        for date, group in factor_df.groupby(level=1):
            for factor in factor_df.columns:
                values = group[factor].dropna()
                if len(values) < 2:
                    continue
                
                mean = values.mean()
                std = values.std()
                if std > 0:
                    standardized = (values - mean) / std
                    result_df.loc[group.index, factor] = standardized
        
        return result_df
    
    def verify_orthogonality(self, factor_df: pd.DataFrame, 
                            threshold: float = 0.1) -> bool:
        """
        验证因子正交性
        
        Args:
            factor_df: 因子数据
            threshold: 相关系数阈值，超过则认为不正交
            
        Returns:
            是否通过正交性检验
        """
        print("验证因子正交性...")
        # 计算相关系数矩阵
        corr_matrix = factor_df.corr().abs()
        
        # 检查非对角元素
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        
        print(f"最大相关系数: {max_corr:.4f}")
        
        if max_corr > threshold:
            print(f"警告：存在相关系数超过阈值{threshold}的因子对")
            # 打印高相关因子对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            for f1, f2, corr in high_corr_pairs[:5]:  # 只显示前5个
                print(f"  {f1} - {f2}: {corr:.4f}")
            return False
        
        print("正交性检验通过")
        return True
    
    def merge_industry_factors(self, style_factors: pd.DataFrame,
                               industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并风格因子和行业因子
        
        Args:
            style_factors: 风格因子数据（已预处理）
            industry_df: 行业数据，index=(instrument, datetime), 值为行业代码
            
        Returns:
            合并后的因子暴露矩阵
        """
        print("合并行业因子...")
        # 创建行业虚拟变量（one-hot编码）
        industry_codes = industry_df.iloc[:, 0].astype(str)
        industry_dummies = pd.get_dummies(industry_codes, prefix='ind')
        
        # 重命名列为行业名称
        industry_name_map = {f"ind_{code}": name 
                            for code, name in INDUSTRY_MAPPING.items()}
        industry_dummies = industry_dummies.rename(columns=industry_name_map)
        
        # 合并风格因子和行业因子
        merged = style_factors.join(industry_dummies, how='inner')
        
        print(f"合并完成，共{len(style_factors.columns)}个风格因子 + "
              f"{len(industry_dummies.columns)}个行业因子 = "
              f"{len(merged.columns)}个因子")
        
        return merged
    
    def build_exposure_matrix(self, raw_data: pd.DataFrame,
                             industry_df: pd.DataFrame,
                             market_cap_df: pd.DataFrame,
                             save_path: Optional[str] = None,
                             n_jobs: int = 1,
                             output_manager: RiskOutputManager = None
                              ) -> pd.DataFrame:
        """
        构建完整的因子暴露矩阵
        
        执行完整的预处理流程：
        1. 计算原始因子
        2. 去极值
        3. 中性化
        4. 正交化
        5. 标准化
        6. 合并行业因子
        
        Args:
            raw_data: 原始数据
            industry_df: 行业数据
            market_cap_df: 市值数据
            save_path: 保存路径
            n_jobs: 并行进程数
            
        Returns:
            完整的因子暴露矩阵
        """
        print("=" * 60)
        print("开始构建因子暴露矩阵...")
        
        # 1. 计算原始因子
        raw_factors = self.calculate_raw_factors(raw_data, n_jobs=n_jobs)
        output_manager.save_data(raw_factors, 'debug/raw_factors', type='csv')

        '''
        # 2. 去极值
        winsorized = self.winsorize_factors(raw_factors, method='median')
        
        # 3. 中性化（行业、市值）
        neutralized = self.neutralize_factors(winsorized, industry_df, market_cap_df)
        
        # 4. 正交化
        orthogonalized = self.orthogonalize_factors(neutralized)
        
        # 5. 标准化
        standardized = self.standardize_factors(orthogonalized)
        
        # 6. 验证正交性
        self.verify_orthogonality(standardized)
        
        # 7. 合并行业因子
        exposure_matrix = self.merge_industry_factors(standardized, industry_df)
        
        # 保存结果
        if save_path:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            exposure_matrix.to_csv(save_path, encoding='utf-8')
            print(f"因子暴露矩阵已保存至: {save_path}")
        
        if self.cache_dir:
            cache_file = self.cache_dir / 'exposure_matrix.csv'
            exposure_matrix.to_csv(cache_file, encoding='utf-8')
            print(f"已缓存至: {cache_file}")
        
        print("因子暴露矩阵构建完成")
        print("=" * 60)
        
        return exposure_matrix
        '''