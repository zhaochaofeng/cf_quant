"""
Barra CNE6 风险模型 - 每月运行脚本
用于更新模型参数（因子收益率历史、协方差矩阵、特异风险）
"""
import sys
import argparse
import calendar
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qlib
from barra.risk_control.barra_engine import BarraRiskEngine


def init_qlib():
    """初始化qlib，注册PTTM自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri='~/.qlib/qlib_data/custom_data_hfq',
        custom_ops=[PTTM]
    )
    print("Qlib初始化完成（已注册PTTM操作符）")


def subtract_months(end_dt: datetime, months: int) -> datetime:
    """
    从指定日期减去指定月份数
    
    准确处理月份计算，考虑不同月份的天数差异：
    - 3月31日 - 1个月 = 2月28/29日
    - 5月31日 - 1个月 = 4月30日
    
    Args:
        end_dt: 结束日期
        months: 要减去的月份数
        
    Returns:
        计算后的开始日期
    """
    # 计算目标年份和月份
    total_months = end_dt.year * 12 + end_dt.month - 1  # 0-based month index
    target_months = total_months - months
    
    year = target_months // 12
    month = target_months % 12 + 1  # 转换回1-based
    
    # 处理日期：如果目标月份天数不足，取该月最后一天
    max_day = calendar.monthrange(year, month)[1]
    day = min(end_dt.day, max_day)
    
    return datetime(year, month, day)


def run_monthly_update(end_date: str, history_months: int = 120):
    """
    运行月度模型更新
    
    Args:
        end_date: 数据截止日期
        history_months: 历史数据月数，默认120（10年）
    """
    # 计算开始日期（准确计算月份，考虑不同月份的天数差异）
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = subtract_months(end_dt, history_months)
    start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"\n{'='*70}")
    print(f"Barra CNE6 月度模型更新")
    print(f"数据区间: {start_date} 至 {end_date}")
    print(f"{'='*70}\n")
    
    # 初始化引擎
    engine = BarraRiskEngine(
        calc_date=end_date,
        portfolio_input='random',  # 月频更新不需要组合
        output_dir='barra/risk_control/output',
        n_jobs=4  # 并行计算
    )
    
    # 运行月频更新
    engine.run_monthly_update(start_date, end_date)
    
    # TODO: 保存模型数据到文件，供日频计算使用
    
    print(f"\n{'='*70}")
    print("月度模型更新完成")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Barra CNE6 月度模型更新')
    parser.add_argument('--end-date', type=str, required=True,
                       help='数据截止日期，格式YYYY-MM-DD')
    parser.add_argument('--history-months', type=int, default=120,
                       help='历史数据月数，默认120')
    
    args = parser.parse_args()
    
    # 初始化qlib
    init_qlib()
    
    # 运行月度更新
    run_monthly_update(
        end_date=args.end_date,
        history_months=args.history_months
    )


if __name__ == '__main__':
    main()
