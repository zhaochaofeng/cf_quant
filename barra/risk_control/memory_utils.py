"""
内存监控和优化工具模块
提供内存使用监控、警告和优化辅助功能
"""
import gc
import psutil
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Callable, Any
from functools import wraps


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, threshold_gb: float = 6.0, warning_gb: float = 7.0):
        """
        初始化内存监控器
        
        Args:
            threshold_gb: 内存阈值（GB），超过则触发优化
            warning_gb: 警告阈值（GB），超过则发出警告
        """
        self.threshold_gb = threshold_gb
        self.warning_gb = warning_gb
        self.process = psutil.Process()
        
    def get_memory_usage_gb(self) -> float:
        """获取当前内存使用（GB）"""
        return self.process.memory_info().rss / (1024 ** 3)
    
    def check_memory(self, context: str = "") -> dict:
        """
        检查内存使用情况
        
        Args:
            context: 上下文描述
            
        Returns:
            内存使用信息字典
        """
        memory_gb = self.get_memory_usage_gb()
        system_memory = psutil.virtual_memory()
        
        info = {
            'process_memory_gb': memory_gb,
            'system_total_gb': system_memory.total / (1024 ** 3),
            'system_available_gb': system_memory.available / (1024 ** 3),
            'system_percent': system_memory.percent,
            'context': context
        }
        
        # 发出警告
        if memory_gb > self.warning_gb:
            warnings.warn(
                f"【内存警告】{context}: 进程内存使用 {memory_gb:.2f}GB "
                f"超过警告阈值 {self.warning_gb}GB",
                ResourceWarning
            )
        elif memory_gb > self.threshold_gb:
            warnings.warn(
                f"【内存提示】{context}: 进程内存使用 {memory_gb:.2f}GB "
                f"接近阈值 {self.threshold_gb}GB，建议优化",
                UserWarning
            )
        
        return info
    
    def print_memory_status(self, context: str = ""):
        """打印内存状态"""
        info = self.check_memory(context)
        print(f"\n【内存状态】{context}")
        print(f"  进程内存: {info['process_memory_gb']:.2f} GB")
        print(f"  系统可用: {info['system_available_gb']:.2f} GB / {info['system_total_gb']:.2f} GB")
        print(f"  系统使用率: {info['system_percent']:.1f}%")


def optimize_memory():
    """强制垃圾回收并优化内存"""
    gc.collect()
    # 尝试释放内存给操作系统（仅在某些Python实现中有效）
    if hasattr(gc, 'freeze'):
        gc.freeze()


def convert_to_float32(df: pd.DataFrame, exclude_cols: Optional[list] = None) -> pd.DataFrame:
    """
    将float64列转换为float32以节省内存
    
    Args:
        df: 输入DataFrame
        exclude_cols: 不排除转换的列名列表
        
    Returns:
        转换后的DataFrame
    """
    exclude_cols = exclude_cols or []
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        if col in exclude_cols:
            continue
        if df_optimized[col].dtype == np.float64:
            df_optimized[col] = df_optimized[col].astype(np.float32)
    
    return df_optimized


def optimize_dataframe_memory(df: pd.DataFrame, 
                              categorical_cols: Optional[list] = None,
                              exclude_float_cols: Optional[list] = None) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    
    Args:
        df: 输入DataFrame
        categorical_cols: 应转换为category类型的列
        exclude_float_cols: 不转换为float32的列
        
    Returns:
        优化后的DataFrame
    """
    df_optimized = df.copy()
    
    # 转换float64为float32
    exclude_float_cols = exclude_float_cols or []
    for col in df_optimized.columns:
        if col in exclude_float_cols:
            continue
        if df_optimized[col].dtype == np.float64:
            df_optimized[col] = df_optimized[col].astype(np.float32)
    
    # 转换指定列为category类型
    if categorical_cols:
        for col in categorical_cols:
            if col in df_optimized.columns:
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


def chunk_list(data: list, chunk_size: int):
    """
    将列表分块处理
    
    Args:
        data: 输入列表
        chunk_size: 每块大小
        
    Yields:
        数据块
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def chunk_dataframe_generator(df: pd.DataFrame, chunk_size: int, by_date: bool = False):
    """
    将DataFrame分块生成器
    
    Args:
        df: 输入DataFrame
        chunk_size: 每块行数或日期数
        by_date: 是否按日期分块
        
    Yields:
        数据块DataFrame
    """
    if by_date and isinstance(df.index, pd.MultiIndex):
        # 按日期分块
        dates = df.index.get_level_values(1).unique()
        for i in range(0, len(dates), chunk_size):
            date_chunk = dates[i:i + chunk_size]
            mask = df.index.get_level_values(1).isin(date_chunk)
            yield df[mask]
    else:
        # 按行分块
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]


def memory_efficient_concat(dfs: list, clear_intermediate: bool = True) -> pd.DataFrame:
    """
    内存高效的DataFrame拼接
    
    Args:
        dfs: DataFrame列表
        clear_intermediate: 是否清空中间结果
        
    Returns:
        拼接后的DataFrame
    """
    result = pd.concat(dfs, ignore_index=False)
    
    if clear_intermediate:
        # 清空中间DataFrame释放内存
        for df in dfs:
            del df
        gc.collect()
    
    return result


def clear_variables(*variables):
    """
    清除变量并强制垃圾回收
    
    Args:
        *variables: 要清除的变量
    """
    for var in variables:
        del var
    gc.collect()


def monitor_memory_usage(func: Callable) -> Callable:
    """
    装饰器：监控函数内存使用情况
    
    Args:
        func: 被装饰的函数
        
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        monitor.print_memory_status(f"进入 {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
        finally:
            monitor.print_memory_status(f"退出 {func.__name__}")
        
        return result
    
    return wrapper


def estimate_dataframe_memory(df: pd.DataFrame) -> dict:
    """
    估算DataFrame内存使用
    
    Args:
        df: 输入DataFrame
        
    Returns:
        内存使用信息字典
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    info = {
        'total_mb': memory_bytes / (1024 ** 2),
        'total_gb': memory_bytes / (1024 ** 3),
        'columns': {}
    }
    
    for col in df.columns:
        col_memory = df[col].memory_usage(deep=True) / (1024 ** 2)
        info['columns'][col] = {
            'dtype': str(df[col].dtype),
            'memory_mb': col_memory
        }
    
    return info


def suggest_workers_by_memory(max_workers: int = 8, 
                              memory_per_worker_gb: float = 1.0,
                              reserve_memory_gb: float = 2.0) -> int:
    """
    根据可用内存建议并行工作进程数
    
    Args:
        max_workers: 最大工作进程数
        memory_per_worker_gb: 每个工作进程预估内存（GB）
        reserve_memory_gb: 保留内存（GB）
        
    Returns:
        建议的工作进程数
    """
    system_memory = psutil.virtual_memory()
    available_gb = system_memory.available / (1024 ** 3)
    usable_gb = available_gb - reserve_memory_gb
    
    suggested_workers = max(1, int(usable_gb / memory_per_worker_gb))
    suggested_workers = min(suggested_workers, max_workers, psutil.cpu_count())
    
    return suggested_workers


# 全局内存监控器实例
memory_monitor = MemoryMonitor()
