"""
横截面回归模块 - 内存优化版本
针对8GB RAM约束优化
"""
import pandas as pd
import numpy as np
import gc
import statsmodels.api as sm
from typing import Tuple, Optional, Generator
from pathlib import Path

from .memory_utils import (
    MemoryMonitor, optimize_memory, convert_to_float32,
    clear_variables, chunk_dataframe_generator
)


class CrossSectionalRegression:
    """横截面回归估计器 - 内存优化版本"""
    
    def __init__(self, weight_type: str = 'sqrt_market_cap',
                 memory_threshold_gb: float = 6.0):
        """
        初始化横截面回归器
        
        Args:
            weight_type: 权重类型，默认'市值平方根'
            memory_threshold_gb: 内存阈值（GB）
        """
        self.weight_type = weight_type
        self.factor_returns = {}  # 存储各期因子收益率
        self.residuals = {}       # 存储各期残差
        self.memory_monitor = MemoryMonitor(threshold_gb=memory_threshold_gb)
        
    def calculate_weights(self, market_cap: pd.Series) -> pd.Series:
        """
        计算回归权重（市值平方根）
        
        Args:
            market_cap: 市值序列
            
        Returns:
            权重序列（float32）
        """
        weights = np.sqrt(market_cap).astype(np.float32)
        return weights
    
    def fit(self, date: str, returns: pd.Series, 
            exposure: pd.DataFrame, market_cap: pd.Series) -> dict:
        """
        单期横截面回归 - 内存优化版本
        
        模型：r_t = X_t * b_t + u_t
        
        Args:
            date: 日期
            returns: 股票收益率序列
            exposure: 因子暴露矩阵
            market_cap: 市值序列
            
        Returns:
            回归结果字典
        """
        # 转换数据类型以节省内存
        returns = returns.astype(np.float32)
        exposure = convert_to_float32(exposure)
        market_cap = market_cap.astype(np.float32)
        
        # 对齐数据
        common_index = returns.index.intersection(exposure.index).intersection(market_cap.index)
        r = returns.loc[common_index]
        X = exposure.loc[common_index]
        mv = market_cap.loc[common_index]
        
        # 处理缺失值
        valid_mask = r.notna() & X.notna().all(axis=1) & mv.notna()
        r = r[valid_mask]
        X = X[valid_mask]
        mv = mv[valid_mask]
        
        if len(r) == 0 or X.shape[1] == 0:
            print(f"{date}: 无有效数据，跳过回归")
            return None
        
        # 计算权重
        weights = self.calculate_weights(mv)
        
        # 加权最小二乘回归
        try:
            model = sm.WLS(r, X, weights=weights)
            results = model.fit()
        except Exception as e:
            print(f"{date}: 回归失败 - {str(e)}")
            return None
        
        # 提取结果（转换为float32）
        factor_returns = results.params.astype(np.float32)
        residuals = results.resid.astype(np.float32)
        
        # 保存结果
        self.factor_returns[date] = factor_returns
        self.residuals[date] = residuals
        
        # 清理中间变量
        clear_variables(r, X, mv, weights, results)
        
        return {
            'date': date,
            'factor_returns': factor_returns,
            'residuals': residuals,
            'r_squared': float(results.rsquared),
            'adj_r_squared': float(results.rsquared_adj),
            'n_obs': int(results.nobs),
            'f_statistic': float(results.fvalue) if results.fvalue else None,
            'f_pvalue': float(results.f_pvalue) if results.f_pvalue else None,
        }
    
    def fit_multi_periods(self, returns_df: pd.DataFrame,
                         exposure_df: pd.DataFrame,
                         market_cap_df: pd.DataFrame,
                         dates_per_batch: int = 10) -> pd.DataFrame:
        """
        多期横截面回归 - 分批处理版本
        
        Args:
            returns_df: 收益率数据
            exposure_df: 因子暴露数据
            market_cap_df: 市值数据
            dates_per_batch: 每批处理日期数
            
        Returns:
            因子收益率矩阵
        """
        print("开始多期横截面回归（内存优化模式）...")
        self.memory_monitor.print_memory_status("回归开始前")
        
        # 转换数据类型
        returns_df = convert_to_float32(returns_df)
        exposure_df = convert_to_float32(exposure_df)
        market_cap_df = convert_to_float32(market_cap_df)
        
        # 获取所有日期
        dates = returns_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        print(f"总日期数: {len(dates)}, 批次数: {total_batches}")
        
        results_list = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            batch_num = batch_idx // dates_per_batch + 1
            print(f"\n处理日期批次 {batch_num}/{total_batches} ({len(date_batch)}天)...")
            
            for date in date_batch:
                # 提取当期数据
                try:
                    r = returns_df.xs(date, level=1).iloc[:, 0]
                    X = exposure_df.xs(date, level=1)
                    mv = market_cap_df.xs(date, level=1).iloc[:, 0]
                    
                    result = self.fit(str(date), r, X, mv)
                    if result:
                        results_list.append(result)
                    
                    # 清理内存
                    clear_variables(r, X, mv)
                    
                except Exception as e:
                    print(f"{date}: 数据处理失败 - {str(e)}")
                    continue
            
            # 每批次后检查和优化内存
            if batch_num % 5 == 0:
                self.memory_monitor.check_memory(f"批次 {batch_num} 完成")
                optimize_memory()
        
        # 构建因子收益率DataFrame
        if results_list:
            factor_returns_df = pd.DataFrame(
                {r['date']: r['factor_returns'] for r in results_list}
            ).T
            factor_returns_df.index.name = 'date'
            # 转换类型
            factor_returns_df = convert_to_float32(factor_returns_df)
        else:
            factor_returns_df = pd.DataFrame()
        
        self.memory_monitor.print_memory_status("回归完成")
        print(f"横截面回归完成，共{len(results_list)}期")
        
        return factor_returns_df
    
    def fit_multi_periods_incremental(self, returns_df: pd.DataFrame,
                                     exposure_df: pd.DataFrame,
                                     market_cap_df: pd.DataFrame,
                                     output_dir: str,
                                     dates_per_batch: int = 10) -> pd.DataFrame:
        """
        增量多期横截面回归 - 将中间结果保存到磁盘
        
        Args:
            returns_df: 收益率数据
            exposure_df: 因子暴露数据
            market_cap_df: 市值数据
            output_dir: 输出目录
            dates_per_batch: 每批处理日期数
            
        Returns:
            因子收益率矩阵
        """
        print("开始增量多期横截面回归（磁盘缓存模式）...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 转换数据类型
        returns_df = convert_to_float32(returns_df)
        exposure_df = convert_to_float32(exposure_df)
        market_cap_df = convert_to_float32(market_cap_df)
        
        # 获取所有日期
        dates = returns_df.index.get_level_values(1).unique()
        total_batches = (len(dates) + dates_per_batch - 1) // dates_per_batch
        
        factor_returns_files = []
        residuals_files = []
        
        for batch_idx in range(0, len(dates), dates_per_batch):
            date_batch = dates[batch_idx:batch_idx + dates_per_batch]
            batch_num = batch_idx // dates_per_batch + 1
            print(f"\n处理日期批次 {batch_num}/{total_batches} ({len(date_batch)}天)...")
            
            batch_factor_returns = {}
            batch_residuals = {}
            
            for date in date_batch:
                try:
                    r = returns_df.xs(date, level=1).iloc[:, 0]
                    X = exposure_df.xs(date, level=1)
                    mv = market_cap_df.xs(date, level=1).iloc[:, 0]
                    
                    result = self.fit(str(date), r, X, mv)
                    if result:
                        batch_factor_returns[result['date']] = result['factor_returns']
                        batch_residuals[result['date']] = result['residuals']
                    
                    clear_variables(r, X, mv)
                    
                except Exception as e:
                    print(f"{date}: 数据处理失败 - {str(e)}")
                    continue
            
            # 保存批次结果到磁盘
            if batch_factor_returns:
                fr_batch_df = pd.DataFrame(batch_factor_returns).T
                fr_batch_df = convert_to_float32(fr_batch_df)
                fr_file = output_path / f"factor_returns_batch_{batch_num:04d}.csv"
                fr_batch_df.to_csv(fr_file, encoding='utf-8')
                factor_returns_files.append(fr_file)
                
                # 保存残差
                resid_batch_df = pd.DataFrame(batch_residuals).T
                resid_batch_df = convert_to_float32(resid_batch_df)
                resid_file = output_path / f"residuals_batch_{batch_num:04d}.csv"
                resid_batch_df.to_csv(resid_file, encoding='utf-8')
                residuals_files.append(resid_file)
                
                print(f"批次 {batch_num} 结果已保存")
                
                # 清理内存
                clear_variables(batch_factor_returns, batch_residuals,
                              fr_batch_df, resid_batch_df)
                optimize_memory()
        
        # 从磁盘合并所有结果
        print("\n从磁盘合并所有结果...")
        fr_chunks = []
        for file in factor_returns_files:
            chunk = pd.read_csv(file, index_col=0, encoding='utf-8')
            chunk = convert_to_float32(chunk)
            fr_chunks.append(chunk)
        
        factor_returns_df = pd.concat(fr_chunks, ignore_index=False)
        factor_returns_df = convert_to_float32(factor_returns_df)
        factor_returns_df.index.name = 'date'
        
        # 重新加载残差到内存
        self.residuals = {}
        for file in residuals_files:
            resid_chunk = pd.read_csv(file, index_col=0, encoding='utf-8')
            for date in resid_chunk.index:
                self.residuals[date] = resid_chunk.loc[date].astype(np.float32)
        
        self.memory_monitor.print_memory_status("增量回归完成")
        print(f"横截面回归完成")
        
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
        df = convert_to_float32(df)
        return df
    
    def get_residuals(self) -> pd.DataFrame:
        """
        获取所有期的残差
        
        Returns:
            DataFrame, index=(instrument, date), columns=['residual']
        """
        if not self.residuals:
            return pd.DataFrame()
        
        # 使用生成器分批构建DataFrame以节省内存
        resid_list = []
        for date, resid in self.residuals.items():
            temp_df = pd.DataFrame({
                'instrument': resid.index,
                'date': date,
                'residual': resid.values.astype(np.float32)
            })
            resid_list.append(temp_df)
        
        df = pd.concat(resid_list, ignore_index=True)
        df = df.set_index(['instrument', 'date'])
        df = convert_to_float32(df)
        
        # 清理中间列表
        clear_variables(resid_list)
        
        return df
    
    def save_residuals_incremental(self, output_file: str, chunk_size: int = 100):
        """
        增量保存残差到文件 - 避免内存溢出
        
        Args:
            output_file: 输出文件路径
            chunk_size: 每块处理的日期数
        """
        if not self.residuals:
            print("无残差数据可保存")
            return
        
        print(f"增量保存残差到: {output_file}")
        
        dates = list(self.residuals.keys())
        total_chunks = (len(dates) + chunk_size - 1) // chunk_size
        
        # 写入CSV头部
        first_batch = True
        
        for i in range(0, len(dates), chunk_size):
            chunk_dates = dates[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            print(f"保存残差块 {chunk_num}/{total_chunks}...")
            
            resid_list = []
            for date in chunk_dates:
                resid = self.residuals[date]
                temp_df = pd.DataFrame({
                    'instrument': resid.index,
                    'date': date,
                    'residual': resid.values.astype(np.float32)
                })
                resid_list.append(temp_df)
            
            chunk_df = pd.concat(resid_list, ignore_index=True)
            
            # 追加写入文件
            chunk_df.to_csv(output_file, 
                          mode='w' if first_batch else 'a',
                          header=first_batch,
                          index=False,
                          encoding='utf-8')
            
            first_batch = False
            clear_variables(resid_list, chunk_df)
        
        print("残差保存完成")
    
    def calculate_residual_variance(self, date: str) -> pd.Series:
        """
        计算特异方差
        
        Args:
            date: 日期
            
        Returns:
            特异方差序列（float32）
        """
        if date not in self.residuals:
            return pd.Series(dtype=np.float32)
        
        residuals = self.residuals[date]
        specific_var = (residuals ** 2).astype(np.float32)
        
        return specific_var
    
    def clear_memory(self):
        """清理内存中的结果数据"""
        self.factor_returns.clear()
        self.residuals.clear()
        optimize_memory()
        print("横截面回归器内存已清理")
