"""
因子暴露矩阵构建模块 - 内存优化版本
针对8GB RAM约束优化：
1. 使用float32替代float64
2. 批量/分块处理数据
3. 使用生成器代替列表
4. 及时释放内存
5. 使用category类型
6. 逐日处理
"""
import pandas as pd
import numpy as np
import gc
import statsmodels.api as sm
from typing import List, Dict, Optional, Generator, Tuple
from pathlib import Path
import sys
import warnings

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import winsorize as utils_winsorize, standardize as utils_standardize
from utils.multiprocess import multiprocessing_wrapper
from .memory_utils import (
    MemoryMonitor, optimize_memory, convert_to_float32,
    optimize_dataframe_memory, chunk_list, chunk_dataframe_generator,
    memory_efficient_concat, clear_variables, monitor_memory_usage,
    estimate_dataframe_memory, suggest_workers_by_memory
)
from data.factor import (
    LNCAP, MIDCAP,
    BETA, HSIGMA, DASTD, CMRA,
    STOM, STOQ, STOA, ATVR,
    STREV, SEASON, INDMOM, RSTR, HALPHA,
    MLEV, BLEV, DTOA,
    VSAL, VERN, VFLO,
    ABS, ACF,
    ATO, GP, GPM, ROA,
    AGRO, IGRO, CXGRO,
    BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA,
    EGRO, SGRO,
)

from .config import STYLE_FACTOR_LIST, INDUSTRY_CODES, INDUSTRY_MAPPING, MODEL_PARAMS


# 因子计算函数字典
FACTOR_FUNCTIONS = {
    'LNCAP': LNCAP, 'MIDCAP': MIDCAP,
    'BETA': BETA, 'HSIGMA': HSIGMA, 'DASTD': DASTD, 'CMRA': CMRA,
    'STOM': STOM, 'STOQ': STOQ, 'STOA': STOA, 'ATVR': ATVR,
    'STREV': STREV, 'SEASON': SEASON, 'INDMOM': INDMOM, 'RSTR': RSTR, 'HALPHA': HALPHA,
    'MLEV': MLEV, 'BLEV': BLEV, 'DTOA': DTOA,
    'VSAL': VSAL, 'VERN': VERN, 'VFLO': VFLO,
    'ABS': ABS, 'ACF': ACF,
    'ATO': ATO, 'GP': GP, 'GPM': GPM, 'ROA': ROA,
    'AGRO': AGRO, 'IGRO': IGRO, 'CXGRO': CXGRO,
    'BTOP': BTOP, 'ETOP': ETOP, 'CETOP': CETOP, 'EM': EM,
    'LTRSTR': LTRSTR, 'LTHALPHA': LTHALPHA,
    'EGRO': EGRO, 'SGRO': SGRO,
}

# 批处理配置
BATCH_CONFIG = {
    'stocks_per_batch': 100,  # 每批处理股票数
    'dates_per_batch': 10,    # 每批处理日期数
    'float_dtype': np.float32,
}


class FactorExposureBuilder:
    """因子暴露矩阵构建器 - 内存优化版本"""
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 memory_threshold_gb: float = 6.0):
        """
        初始化因子暴露构建器
        
        Args:
            cache_dir: 缓存目录路径，用于保存中间结果
            memory_threshold_gb: 内存阈值（GB）
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_monitor = MemoryMonitor(threshold_gb=memory_threshold_gb)
        self._check_memory = True
        
    def _check_and_optimize_memory(self, context: str):
        """检查内存并优化"""
        if self._check_memory:
            info = self.memory_monitor.check_memory(context)
            if info['process_memory_gb'] > self.memory_monitor.threshold_gb:
                optimize_memory()
                self.memory_monitor.print_memory_status(f"优化后 - {context}")
    
    def calculate_raw_factors(self, raw_data: pd.DataFrame, 
                             n_jobs: int = 1,
                             batch_size: int = None) -> pd.DataFrame:
        """
        计算原始CNE6因子值 - 分批处理版本
        
        Args:
            raw_data: 原始数据DataFrame
            n_jobs: 并行进程数（根据内存自动调整）
            batch_size: 批处理股票数，None则自动计算
            
        Returns:
            DataFrame, index=(instrument, datetime), columns=因子名称
        """
        print("开始计算原始因子值（内存优化模式）...")
        self.memory_monitor.print_memory_status("计算开始前")
        
        raw_data = raw_data.sort_index()
        
        # 自动调整并行进程数
        if n_jobs > 1:
            n_jobs = suggest_workers_by_memory(
                max_workers=n_jobs,
                memory_per_worker_gb=0.8,
                reserve_memory_gb=2.0
            )
            print(f"根据内存自动调整并行进程数为: {n_jobs}")
        
        # 确定批大小
        if batch_size is None:
            batch_size = BATCH_CONFIG['stocks_per_batch']
        
        # 获取所有股票和日期
        all_instruments = raw_data.index.get_level_values(0).unique()
        all_dates = raw_data.index.get_level_values(1).unique()
        
        print(f"总股票数: {len(all_instruments)}, 总日期数: {len(all_dates)}")
        print(f"批处理大小: {batch_size}只股票/批")
        
        # 分批计算
        factor_results = {}
        total_batches = (len(all_instruments) + batch_size - 1) // batch_size
        
        for batch_idx, instrument_batch in enumerate(chunk_list(list(all_instruments), batch_size)):
            print(f"\n处理第 {batch_idx + 1}/{total_batches} 批股票 ({len(instrument_batch)}只)...")
            
            # 提取当前批次数据
            batch_mask = raw_data.index.get_level_values(0).isin(instrument_batch)
            batch_data = raw_data[batch_mask]
            
            # 转换float64为float32以节省内存
            batch_data = convert_to_float32(batch_data)
            
            # 计算该批次的因子
            batch_results = self._calculate_factors_for_batch(
                batch_data, n_jobs=n_jobs
            )
            
            # 合并结果
            for factor_name, series in batch_results.items():
                if factor_name not in factor_results:
                    factor_results[factor_name] = []
                factor_results[factor_name].append(series)
            
            # 释放批次数据内存
            clear_variables(batch_data, batch_results)
            self._check_and_optimize_memory(f"批次 {batch_idx + 1} 完成后")
        
        # 合并所有批次的因子结果
        print("\n合并所有批次结果...")
        factor_df = pd.DataFrame(index=raw_data.index)
        
        for factor_name, series_list in factor_results.items():
            # 使用内存高效的拼接
            factor_df[factor_name] = pd.concat(series_list, ignore_index=False)
            clear_variables(series_list)
        
        # 转换结果类型
        factor_df = convert_to_float32(factor_df)
        
        self.memory_monitor.print_memory_status("原始因子计算完成")
        print(f"原始因子计算完成，共{len(factor_df.columns)}个因子")
        
        return factor_df
    
    def _calculate_factors_for_batch(self, batch_data: pd.DataFrame,
                                     n_jobs: int = 1) -> Dict[str, pd.Series]:
        """
        计算单批数据的因子
        
        Args:
            batch_data: 批次数据
            n_jobs: 并行进程数
            
        Returns:
            因子结果字典
        """
        batch_results = {}
        
        if n_jobs > 1:
            # 并行计算
            func_calls = []
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    func_calls.append((
                        self._compute_single_factor,
                        (batch_data, factor_name, FACTOR_FUNCTIONS[factor_name])
                    ))
            
            results = multiprocessing_wrapper(func_calls, n=n_jobs)
            for factor_name, result in results:
                if result is not None:
                    batch_results[factor_name] = result
        else:
            # 串行计算（内存更友好）
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    _, result = self._compute_single_factor(
                        batch_data, factor_name, FACTOR_FUNCTIONS[factor_name]
                    )
                    if result is not None:
                        batch_results[factor_name] = result
        
        return batch_results
    
    def _compute_single_factor(self, raw_data: pd.DataFrame, 
                               factor_name: str, 
                               factor_func) -> tuple:
        """计算单个因子（用于并行）"""
        try:
            result = factor_func(raw_data)
            if result is not None and not result.empty:
                series = result.iloc[:, 0]
                # 转换为float32
                series = series.astype(np.float32)
                return factor_name, series
        except Exception as e:
            print(f"因子{factor_name}计算失败: {str(e)}")
        return factor_name, None
    
    def winsorize_factors(self, factor_df: pd.DataFrame, 
                         method: str = 'median',
                         dates_per_batch: int = None) -> pd.DataFrame:
        """
        因子去极值处理 - 逐日分批版本
        
        Args:
            factor_df: 因子数据
            method: 去极值方法
            dates_per_batch: 每批处理日期数，None则使用默认配置
            
        Returns:
            去极值后的因子数据
        """
        print("进行中位数去极值（逐日处理）...")
        self.memory_monitor.print_memory_status("去极值开始前")
        
        if dates_per_batch is None:
            dates_per_batch = BATCH_CONFIG['dates_per_batch']
        
        # 转换数据类型
        factor_df = convert_to_float32(factor_df)
        
        # 获取所有日期并分批
        dates = factor_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        result_chunks = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            print(f"处理日期批次 {batch_idx//dates_per_batch + 1}/{total_batches}...")
            
            # 提取当前批次数据
            batch_mask = factor_df.index.get_level_values(1).isin(date_batch)
            batch_df = factor_df[batch_mask].copy()
            
            # 处理该批次
            batch_result = self._winsorize_batch(batch_df, method)
            result_chunks.append(batch_result)
            
            # 释放内存
            clear_variables(batch_df, batch_result)
            self._check_and_optimize_memory(f"去极值批次 {batch_idx//dates_per_batch + 1} 完成后")
        
        # 合并结果
        print("合并去极值结果...")
        result_df = memory_efficient_concat(result_chunks)
        result_df = convert_to_float32(result_df)
        
        self.memory_monitor.print_memory_status("去极值完成")
        return result_df
    
    def _winsorize_batch(self, batch_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """处理单批数据的去极值"""
        result_df = pd.DataFrame(index=batch_df.index)
        
        # 按日期分组处理
        for date, group in batch_df.groupby(level=1):
            for factor in batch_df.columns:
                values = group[factor].dropna()
                if len(values) == 0:
                    continue
                
                if method == 'median':
                    median = values.median()
                    mad = np.median(np.abs(values - median))
                    lower_bound = median - 5 * 1.4826 * mad
                    upper_bound = median + 5 * 1.4826 * mad
                else:
                    lower_bound = values.quantile(0.01)
                    upper_bound = values.quantile(0.99)
                
                clipped = values.clip(lower=lower_bound, upper=upper_bound)
                result_df.loc[group.index, factor] = clipped
        
        return result_df
    
    def neutralize_factors(self, factor_df: pd.DataFrame, 
                          industry_df: pd.DataFrame,
                          market_cap_df: pd.DataFrame,
                          dates_per_batch: int = None) -> pd.DataFrame:
        """
        行业/市值中性化 - 逐日分批版本
        
        Args:
            factor_df: 因子数据
            industry_df: 行业数据
            market_cap_df: 市值数据
            dates_per_batch: 每批处理日期数
            
        Returns:
            中性化后的因子数据
        """
        print("进行行业/市值中性化（逐日处理）...")
        self.memory_monitor.print_memory_status("中性化开始前")
        
        if dates_per_batch is None:
            dates_per_batch = BATCH_CONFIG['dates_per_batch']
        
        # 转换数据类型
        factor_df = convert_to_float32(factor_df)
        market_cap_df = convert_to_float32(market_cap_df)
        
        # 优化行业数据类型
        if industry_df.iloc[:, 0].dtype != 'category':
            industry_df = industry_df.copy()
            industry_df.iloc[:, 0] = industry_df.iloc[:, 0].astype('category')
        
        # 获取所有日期
        dates = factor_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        result_chunks = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            print(f"处理日期批次 {batch_idx//dates_per_batch + 1}/{total_batches}...")
            
            # 提取当前批次数据
            batch_mask = factor_df.index.get_level_values(1).isin(date_batch)
            batch_factor = factor_df[batch_mask]
            batch_industry = industry_df[industry_df.index.get_level_values(1).isin(date_batch)]
            batch_cap = market_cap_df[market_cap_df.index.get_level_values(1).isin(date_batch)]
            
            # 处理该批次
            batch_result = self._neutralize_batch(batch_factor, batch_industry, batch_cap)
            result_chunks.append(batch_result)
            
            # 释放内存
            clear_variables(batch_factor, batch_industry, batch_cap, batch_result)
            self._check_and_optimize_memory(f"中性化批次 {batch_idx//dates_per_batch + 1} 完成后")
        
        # 合并结果
        print("合并中性化结果...")
        result_df = memory_efficient_concat(result_chunks)
        result_df = convert_to_float32(result_df)
        
        self.memory_monitor.print_memory_status("中性化完成")
        return result_df
    
    def _neutralize_batch(self, factor_df: pd.DataFrame,
                         industry_df: pd.DataFrame,
                         market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """处理单批数据的中性化"""
        result_df = pd.DataFrame(index=factor_df.index)
        
        # 准备行业虚拟变量（使用稀疏矩阵节省内存）
        industry_series = industry_df.iloc[:, 0]
        industry_dummies = pd.get_dummies(industry_series, prefix='ind', sparse=True)
        
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
        """对单个因子进行中性化"""
        y = merged_df[factor_name]
        X_cols = industry_cols + ['circ_mv']
        X = merged_df[X_cols].copy()
        X['circ_mv'] = np.log(X['circ_mv'])
        X = sm.add_constant(X)
        
        weights = np.sqrt(merged_df['circ_mv'])
        
        try:
            model = sm.WLS(y, X, weights=weights, missing='drop')
            results = model.fit()
            
            resid = pd.Series(index=merged_df.index, dtype=np.float32)
            resid.loc[results.resid.index] = results.resid.values.astype(np.float32)
            return resid
        except Exception as e:
            print(f"中性化失败 ({factor_name}): {str(e)}")
            return pd.Series(index=merged_df.index, dtype=np.float32)
    
    def orthogonalize_factors(self, factor_df: pd.DataFrame,
                             factor_order: Optional[List[str]] = None,
                             dates_per_batch: int = None) -> pd.DataFrame:
        """
        因子正交化 - 逐日分批版本
        
        Args:
            factor_df: 因子数据
            factor_order: 因子顺序列表
            dates_per_batch: 每批处理日期数
            
        Returns:
            正交化后的因子数据
        """
        print("进行因子正交化（逐日处理）...")
        self.memory_monitor.print_memory_status("正交化开始前")
        
        if dates_per_batch is None:
            dates_per_batch = BATCH_CONFIG['dates_per_batch']
        
        if factor_order is None:
            factor_order = [f for f in STYLE_FACTOR_LIST if f in factor_df.columns]
        
        # 转换数据类型
        factor_df = convert_to_float32(factor_df)
        
        # 获取所有日期
        dates = factor_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        result_chunks = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            print(f"处理日期批次 {batch_idx//dates_per_batch + 1}/{total_batches}...")
            
            # 提取当前批次数据
            batch_mask = factor_df.index.get_level_values(1).isin(date_batch)
            batch_df = factor_df[batch_mask].copy()
            
            # 处理该批次
            batch_result = self._orthogonalize_batch(batch_df, factor_order)
            result_chunks.append(batch_result)
            
            # 释放内存
            clear_variables(batch_df, batch_result)
            self._check_and_optimize_memory(f"正交化批次 {batch_idx//dates_per_batch + 1} 完成后")
        
        # 合并结果
        print("合并正交化结果...")
        result_df = memory_efficient_concat(result_chunks)
        result_df = convert_to_float32(result_df)
        
        self.memory_monitor.print_memory_status("正交化完成")
        return result_df
    
    def _orthogonalize_batch(self, batch_df: pd.DataFrame,
                            factor_order: List[str]) -> pd.DataFrame:
        """处理单批数据的正交化"""
        result_df = pd.DataFrame(index=batch_df.index)
        
        for i, factor_name in enumerate(factor_order):
            if factor_name not in batch_df.columns:
                continue
            
            if i == 0:
                result_df[factor_name] = batch_df[factor_name]
            else:
                y = batch_df[factor_name]
                X = result_df.iloc[:, :i]
                X = sm.add_constant(X)
                
                try:
                    model = sm.OLS(y, X, missing='drop').fit()
                    resid = pd.Series(index=batch_df.index, dtype=np.float32)
                    resid.loc[model.resid.index] = model.resid.values.astype(np.float32)
                    result_df[factor_name] = resid
                except Exception as e:
                    print(f"正交化失败 ({factor_name}): {str(e)}")
                    result_df[factor_name] = batch_df[factor_name]
        
        return result_df
    
    def standardize_factors(self, factor_df: pd.DataFrame,
                           dates_per_batch: int = None) -> pd.DataFrame:
        """
        因子标准化（Z-Score）- 逐日分批版本
        
        Args:
            factor_df: 因子数据
            dates_per_batch: 每批处理日期数
            
        Returns:
            标准化后的因子数据
        """
        print("进行标准化（逐日处理）...")
        self.memory_monitor.print_memory_status("标准化开始前")
        
        if dates_per_batch is None:
            dates_per_batch = BATCH_CONFIG['dates_per_batch']
        
        # 转换数据类型
        factor_df = convert_to_float32(factor_df)
        
        # 获取所有日期
        dates = factor_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        result_chunks = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            print(f"处理日期批次 {batch_idx//dates_per_batch + 1}/{total_batches}...")
            
            # 提取当前批次数据
            batch_mask = factor_df.index.get_level_values(1).isin(date_batch)
            batch_df = factor_df[batch_mask].copy()
            
            # 处理该批次
            batch_result = self._standardize_batch(batch_df)
            result_chunks.append(batch_result)
            
            # 释放内存
            clear_variables(batch_df, batch_result)
            self._check_and_optimize_memory(f"标准化批次 {batch_idx//dates_per_batch + 1} 完成后")
        
        # 合并结果
        print("合并标准化结果...")
        result_df = memory_efficient_concat(result_chunks)
        result_df = convert_to_float32(result_df)
        
        self.memory_monitor.print_memory_status("标准化完成")
        return result_df
    
    def _standardize_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """处理单批数据的标准化"""
        result_df = pd.DataFrame(index=batch_df.index)
        
        for date, group in batch_df.groupby(level=1):
            for factor in batch_df.columns:
                values = group[factor].dropna()
                if len(values) < 2:
                    continue
                
                mean = values.mean()
                std = values.std()
                if std > 0:
                    standardized = ((values - mean) / std).astype(np.float32)
                    result_df.loc[group.index, factor] = standardized
        
        return result_df
    
    def verify_orthogonality(self, factor_df: pd.DataFrame, 
                            threshold: float = 0.1,
                            sample_size: int = 10000) -> bool:
        """
        验证因子正交性 - 采样版本
        
        Args:
            factor_df: 因子数据
            threshold: 相关系数阈值
            sample_size: 采样大小以节省内存
            
        Returns:
            是否通过正交性检验
        """
        print("验证因子正交性（采样验证）...")
        
        # 采样以节省内存
        if len(factor_df) > sample_size:
            sample_df = factor_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = factor_df
        
        # 转换为float32
        sample_df = convert_to_float32(sample_df)
        
        corr_matrix = sample_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        
        print(f"最大相关系数: {max_corr:.4f}")
        
        if max_corr > threshold:
            print(f"警告：存在相关系数超过阈值{threshold}的因子对")
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            for f1, f2, corr in high_corr_pairs[:5]:
                print(f"  {f1} - {f2}: {corr:.4f}")
            return False
        
        print("正交性检验通过")
        return True
    
    def merge_industry_factors(self, style_factors: pd.DataFrame,
                               industry_df: pd.DataFrame,
                               dates_per_batch: int = None) -> pd.DataFrame:
        """
        合并风格因子和行业因子 - 分批版本
        
        Args:
            style_factors: 风格因子数据
            industry_df: 行业数据
            dates_per_batch: 每批处理日期数
            
        Returns:
            合并后的因子暴露矩阵
        """
        print("合并行业因子（分批处理）...")
        self.memory_monitor.print_memory_status("合并行业因子开始前")
        
        if dates_per_batch is None:
            dates_per_batch = BATCH_CONFIG['dates_per_batch']
        
        # 转换数据类型
        style_factors = convert_to_float32(style_factors)
        
        # 优化行业数据类型
        if industry_df.iloc[:, 0].dtype != 'category':
            industry_df = industry_df.copy()
            industry_df.iloc[:, 0] = industry_df.iloc[:, 0].astype('category')
        
        # 获取所有日期
        dates = style_factors.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        result_chunks = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            print(f"处理日期批次 {batch_idx//dates_per_batch + 1}/{total_batches}...")
            
            # 提取当前批次数据
            batch_mask = style_factors.index.get_level_values(1).isin(date_batch)
            batch_style = style_factors[batch_mask]
            batch_industry = industry_df[industry_df.index.get_level_values(1).isin(date_batch)]
            
            # 创建行业虚拟变量
            industry_codes = batch_industry.iloc[:, 0].astype(str)
            industry_dummies = pd.get_dummies(industry_codes, prefix='ind', sparse=True)
            
            # 重命名列
            industry_name_map = {f"ind_{code}": name 
                                for code, name in INDUSTRY_MAPPING.items()}
            industry_dummies = industry_dummies.rename(columns=industry_name_map)
            
            # 合并
            merged = batch_style.join(industry_dummies, how='inner')
            result_chunks.append(merged)
            
            # 释放内存
            clear_variables(batch_style, batch_industry, industry_dummies, merged)
            self._check_and_optimize_memory(f"合并批次 {batch_idx//dates_per_batch + 1} 完成后")
        
        # 合并结果
        print("合并所有批次结果...")
        result_df = memory_efficient_concat(result_chunks)
        result_df = convert_to_float32(result_df)
        
        self.memory_monitor.print_memory_status("行业因子合并完成")
        print(f"合并完成，共{len(style_factors.columns)}个风格因子 + "
              f"{len(INDUSTRY_MAPPING)}个行业因子 = {len(result_df.columns)}个因子")
        
        return result_df
    
    def build_exposure_matrix(self, raw_data: pd.DataFrame,
                             industry_df: pd.DataFrame,
                             market_cap_df: pd.DataFrame,
                             save_path: Optional[str] = None,
                             n_jobs: int = 1,
                             stock_batch_size: int = None,
                             date_batch_size: int = None) -> pd.DataFrame:
        """
        构建完整的因子暴露矩阵 - 内存优化版本
        
        执行完整的预处理流程：
        1. 计算原始因子（分批）
        2. 去极值（逐日分批）
        3. 中性化（逐日分批）
        4. 正交化（逐日分批）
        5. 标准化（逐日分批）
        6. 合并行业因子（分批）
        
        Args:
            raw_data: 原始数据
            industry_df: 行业数据
            market_cap_df: 市值数据
            save_path: 保存路径
            n_jobs: 并行进程数
            stock_batch_size: 股票批大小
            date_batch_size: 日期批大小
            
        Returns:
            完整的因子暴露矩阵
        """
        print("\n" + "=" * 60)
        print("开始构建因子暴露矩阵（内存优化模式）...")
        print(f"配置: 股票批次={stock_batch_size or BATCH_CONFIG['stocks_per_batch']}, "
              f"日期批次={date_batch_size or BATCH_CONFIG['dates_per_batch']}")
        
        self.memory_monitor.print_memory_status("构建开始前")
        
        # 1. 计算原始因子
        raw_factors = self.calculate_raw_factors(
            raw_data, n_jobs=n_jobs, batch_size=stock_batch_size
        )
        
        # 2. 去极值
        winsorized = self.winsorize_factors(
            raw_factors, method='median', dates_per_batch=date_batch_size
        )
        clear_variables(raw_factors)
        
        # 3. 中性化
        neutralized = self.neutralize_factors(
            winsorized, industry_df, market_cap_df, dates_per_batch=date_batch_size
        )
        clear_variables(winsorized)
        
        # 4. 正交化
        orthogonalized = self.orthogonalize_factors(
            neutralized, dates_per_batch=date_batch_size
        )
        clear_variables(neutralized)
        
        # 5. 标准化
        standardized = self.standardize_factors(
            orthogonalized, dates_per_batch=date_batch_size
        )
        clear_variables(orthogonalized)
        
        # 6. 验证正交性
        self.verify_orthogonality(standardized)
        
        # 7. 合并行业因子
        exposure_matrix = self.merge_industry_factors(
            standardized, industry_df, dates_per_batch=date_batch_size
        )
        clear_variables(standardized)
        
        # 保存结果
        if save_path:
            exposure_matrix.to_csv(save_path, encoding='utf-8')
            print(f"因子暴露矩阵已保存至: {save_path}")
        
        if self.cache_dir:
            cache_file = self.cache_dir / 'exposure_matrix.csv'
            exposure_matrix.to_csv(cache_file, encoding='utf-8')
            print(f"已缓存至: {cache_file}")
        
        self.memory_monitor.print_memory_status("因子暴露矩阵构建完成")
        print("因子暴露矩阵构建完成")
        print("=" * 60)
        
        return exposure_matrix
    
    def build_exposure_matrix_incremental(self, raw_data: pd.DataFrame,
                                         industry_df: pd.DataFrame,
                                         market_cap_df: pd.DataFrame,
                                         output_dir: str,
                                         n_jobs: int = 1,
                                         date_batch_size: int = 5) -> pd.DataFrame:
        """
        增量构建因子暴露矩阵 - 将中间结果保存到磁盘以释放内存
        
        Args:
            raw_data: 原始数据
            industry_df: 行业数据
            market_cap_df: 市值数据
            output_dir: 中间结果输出目录
            n_jobs: 并行进程数
            date_batch_size: 日期批大小
            
        Returns:
            完整的因子暴露矩阵
        """
        print("\n" + "=" * 60)
        print("开始增量构建因子暴露矩阵（磁盘缓存模式）...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有日期
        dates = raw_data.index.get_level_values(1).unique()
        total_batches = (len(dates) + date_batch_size - 1) // date_batch_size
        
        exposure_files = []
        
        for batch_idx in range(0, len(dates), date_batch_size):
            date_batch = dates[batch_idx:batch_idx + date_batch_size]
            batch_num = batch_idx // date_batch_size + 1
            print(f"\n处理日期批次 {batch_num}/{total_batches} ({len(date_batch)}天)...")
            
            # 提取当前批次数据
            batch_mask = raw_data.index.get_level_values(1).isin(date_batch)
            batch_raw = raw_data[batch_mask]
            batch_industry = industry_df[industry_df.index.get_level_values(1).isin(date_batch)]
            batch_cap = market_cap_df[market_cap_df.index.get_level_values(1).isin(date_batch)]
            
            # 处理该批次
            batch_builder = FactorExposureBuilder(memory_threshold_gb=5.0)
            batch_exposure = batch_builder.build_exposure_matrix(
                batch_raw, batch_industry, batch_cap,
                n_jobs=min(n_jobs, 2),  # 限制每批次的并行度
                stock_batch_size=50,
                date_batch_size=date_batch_size
            )
            
            # 保存到磁盘
            batch_file = output_path / f"exposure_batch_{batch_num:04d}.csv"
            batch_exposure.to_csv(batch_file, encoding='utf-8')
            exposure_files.append(batch_file)
            
            print(f"批次 {batch_num} 已保存至: {batch_file}")
            
            # 完全释放内存
            clear_variables(batch_raw, batch_industry, batch_cap, 
                          batch_builder, batch_exposure)
            optimize_memory()
            
            self.memory_monitor.print_memory_status(f"批次 {batch_num} 保存后")
        
        # 从磁盘合并所有批次
        print("\n从磁盘合并所有批次...")
        exposure_chunks = []
        for file in exposure_files:
            chunk = pd.read_csv(file, index_col=[0, 1], encoding='utf-8')
            chunk = convert_to_float32(chunk)
            exposure_chunks.append(chunk)
        
        exposure_matrix = memory_efficient_concat(exposure_chunks)
        
        # 可选：删除中间文件
        # for file in exposure_files:
        #     file.unlink()
        
        self.memory_monitor.print_memory_status("增量构建完成")
        print("因子暴露矩阵增量构建完成")
        print("=" * 60)
        
        return exposure_matrix
