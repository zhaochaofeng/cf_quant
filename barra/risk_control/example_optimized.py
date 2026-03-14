"""
Barra风险模型内存优化版本使用示例
展示如何在8GB RAM环境下高效运行风险模型
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from barra.risk_control import (
    BarraRiskEngine,
    MemoryMonitor,
    suggest_workers_by_memory,
    optimize_memory,
)


def example_basic_usage():
    """
    示例1: 基本用法 - 使用内存优化引擎
    适用于8GB RAM的标准配置
    """
    print("=" * 70)
    print("示例1: 基本用法 - 内存优化模式")
    print("=" * 70)
    
    # 创建内存监控器
    monitor = MemoryMonitor(threshold_gb=6.0)
    monitor.print_memory_status("初始化前")
    
    # 初始化引擎（自动调整并行进程数）
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        portfolio_input='random',  # 随机生成50只股票组合
        market='csi300',
        output_dir='output',
        cache_dir='cache',
        n_jobs=4,  # 最大并行进程数（会根据内存自动调整）
        memory_threshold_gb=6.0,  # 内存阈值
        use_incremental=False     # 不使用磁盘缓存（内存足够时）
    )
    
    monitor.print_memory_status("引擎初始化后")
    
    # 月频更新 - 使用分批处理
    engine.run_monthly_update(
        start_date='2014-03-01',
        end_date='2024-03-01',
        stock_batch_size=100,  # 每批处理100只股票
        date_batch_size=10     # 每批处理10天数据
    )
    
    monitor.print_memory_status("月频更新后")
    
    # 日频风险计算
    risk_results = engine.run_daily_risk()
    
    # 打印风险报告
    engine.print_risk_report()
    
    # 保存结果
    engine.save_results()
    
    # 清理内存
    engine.clear_memory()
    monitor.print_memory_status("清理后")


def example_incremental_mode():
    """
    示例2: 增量模式 - 使用磁盘缓存
    适用于内存严重不足的情况（< 8GB）
    """
    print("\n" + "=" * 70)
    print("示例2: 增量模式 - 磁盘缓存")
    print("=" * 70)
    
    monitor = MemoryMonitor(threshold_gb=5.0)
    
    # 初始化引擎 - 启用增量模式
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        portfolio_input='random',
        output_dir='output',
        memory_threshold_gb=5.0,
        use_incremental=True  # 启用磁盘缓存
    )
    
    print("\n使用增量模式处理大量数据...")
    print("中间结果将保存到磁盘，大幅降低内存使用")
    
    # 月频更新 - 增量模式
    engine.run_monthly_update(
        start_date='2014-03-01',
        end_date='2024-03-01',
        stock_batch_size=50,   # 更小的批大小
        date_batch_size=5      # 更小的日期批
    )
    
    # 日频计算
    engine.run_daily_risk()
    engine.save_results()
    engine.clear_memory()
    
    print("\n增量模式完成，检查磁盘缓存目录: output/incremental_cache/")


def example_custom_batches():
    """
    示例3: 自定义批处理大小
    根据具体硬件配置调整
    """
    print("\n" + "=" * 70)
    print("示例3: 自定义批处理大小")
    print("=" * 70)
    
    # 根据可用内存自动建议配置
    n_jobs = suggest_workers_by_memory(
        max_workers=8,
        memory_per_worker_gb=0.8,
        reserve_memory_gb=2.0
    )
    print(f"根据内存自动建议并行进程数: {n_jobs}")
    
    # 根据内存确定批大小
    import psutil
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    if available_gb > 12:
        stock_batch = 200
        date_batch = 20
        print(f"检测到充足内存({available_gb:.1f}GB)，使用较大批大小")
    elif available_gb > 6:
        stock_batch = 100
        date_batch = 10
        print(f"检测到标准内存({available_gb:.1f}GB)，使用标准批大小")
    else:
        stock_batch = 50
        date_batch = 5
        print(f"检测到有限内存({available_gb:.1f}GB)，使用较小批大小")
    
    print(f"配置: stock_batch={stock_batch}, date_batch={date_batch}, n_jobs={n_jobs}")
    
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        portfolio_input='random',
        output_dir='output',
        n_jobs=n_jobs
    )
    
    engine.run_monthly_update(
        start_date='2019-03-01',  # 使用较短的历史数据
        end_date='2024-03-01',
        stock_batch_size=stock_batch,
        date_batch_size=date_batch
    )
    
    engine.run_daily_risk()
    engine.save_results()


def example_memory_monitoring():
    """
    示例4: 内存监控和优化
    在关键步骤监控内存使用
    """
    print("\n" + "=" * 70)
    print("示例4: 内存监控和优化")
    print("=" * 70)
    
    monitor = MemoryMonitor(threshold_gb=6.0, warning_gb=7.0)
    
    # 初始化
    monitor.print_memory_status("步骤1: 初始化")
    
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        portfolio_input='random',
        output_dir='output'
    )
    
    # 获取内存报告
    mem_info = engine.get_memory_report()
    print(f"\n内存报告: 进程使用 {mem_info['process_memory_gb']:.2f}GB, "
          f"系统可用 {mem_info['system_available_gb']:.2f}GB")
    
    # 手动触发垃圾回收
    print("\n手动触发垃圾回收...")
    optimize_memory()
    monitor.print_memory_status("垃圾回收后")


def example_compare_performance():
    """
    示例5: 性能对比 - 展示内存优化效果
    """
    print("\n" + "=" * 70)
    print("示例5: 内存优化效果对比")
    print("=" * 70)
    
    import pandas as pd
    import numpy as np
    
    # 创建示例数据
    n_stocks = 1000
    n_dates = 100
    n_factors = 38
    
    print(f"\n创建示例数据: {n_stocks}只股票, {n_dates}天, {n_factors}个因子")
    
    # 创建MultiIndex
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    instruments = [f'SH{i:06d}' for i in range(n_stocks)]
    index = pd.MultiIndex.from_product(
        [instruments, dates],
        names=['instrument', 'datetime']
    )
    
    # 比较float64 vs float32内存使用
    print("\n内存使用对比:")
    
    # float64版本
    data_f64 = np.random.randn(len(index), n_factors)
    df_f64 = pd.DataFrame(
        data_f64,
        index=index,
        columns=[f'Factor_{i}' for i in range(n_factors)]
    )
    mem_f64 = df_f64.memory_usage(deep=True).sum() / (1024**2)
    print(f"  float64: {mem_f64:.2f} MB")
    
    # float32版本
    df_f32 = df_f64.astype(np.float32)
    mem_f32 = df_f32.memory_usage(deep=True).sum() / (1024**2)
    print(f"  float32: {mem_f32:.2f} MB")
    print(f"  节省内存: {(mem_f64 - mem_f32):.2f} MB ({(1 - mem_f32/mem_f64)*100:.1f}%)")
    
    # 清理
    del df_f64, df_f32
    optimize_memory()
    print("\n数据已清理")


if __name__ == '__main__':
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("Barra风险模型内存优化版本使用示例")
    print("=" * 70)
    
    # 运行示例
    try:
        example_basic_usage()
    except Exception as e:
        print(f"示例1失败: {e}")
    
    try:
        example_incremental_mode()
    except Exception as e:
        print(f"示例2失败: {e}")
    
    try:
        example_custom_batches()
    except Exception as e:
        print(f"示例3失败: {e}")
    
    try:
        example_memory_monitoring()
    except Exception as e:
        print(f"示例4失败: {e}")
    
    try:
        example_compare_performance()
    except Exception as e:
        print(f"示例5失败: {e}")
    
    print("\n" + "=" * 70)
    print("所有示例运行完成")
    print("=" * 70)
