"""
Barra CNE6 风险模型 - 每日运行脚本
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

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


def run_daily_risk(calc_date: str, portfolio_input='random', 
                  model_data_dir: str = None):
    """
    运行每日风险计算
    
    Args:
        calc_date: 计算日期
        portfolio_input: 投资组合输入
        model_data_dir: 模型数据目录（包含预计算的因子协方差矩阵等）
    """
    print(f"\n{'='*70}")
    print(f"Barra CNE6 每日风险计算")
    print(f"计算日期: {calc_date}")
    print(f"{'='*70}\n")
    
    # 初始化引擎
    engine = BarraRiskEngine(
        calc_date=calc_date,
        portfolio_input=portfolio_input,
        output_dir='barra/risk_control/output',
        n_jobs=1
    )
    
    # 从文件加载预计算的模型数据（如果提供了模型目录）
    if model_data_dir:
        print("\n加载预计算的模型参数（协方差矩阵、特异风险）...")
        success = engine.load_model_data(model_data_dir, calc_date)
        if not success:
            print("警告：无法加载模型数据，将使用默认方式")
        else:
            # 重新计算指定日期的因子暴露
            print(f"\n重新计算 {calc_date} 的因子暴露矩阵...")
            daily_exposure = engine.calculate_daily_exposure(calc_date)
            # 更新引擎中的因子暴露
            engine.factor_exposure = daily_exposure
    
    # 运行日频风险计算
    risk_results = engine.run_daily_risk()
    
    # 保存结果
    saved_files = engine.save_results()
    
    # 打印报告
    engine.print_risk_report()
    
    print(f"\n结果已保存:")
    for key, filepath in saved_files.items():
        print(f"  {key}: {filepath}")
    
    return risk_results


def main():
    parser = argparse.ArgumentParser(description='Barra CNE6 每日风险计算')
    parser.add_argument('--date', type=str, required=True,
                       help='计算日期，格式YYYY-MM-DD')
    parser.add_argument('--portfolio', type=str, default='random',
                       help='投资组合：random(随机) 或 CSV文件路径')
    parser.add_argument('--model-data', type=str, default='barra/risk_control/model_data',
                       help='模型数据目录路径，默认为 barra/risk_control/model_data')
    
    args = parser.parse_args()
    
    # 初始化qlib
    init_qlib()
    
    # 运行风险计算
    run_daily_risk(
        calc_date=args.date,
        portfolio_input=args.portfolio,
        model_data_dir=args.model_data
    )


if __name__ == '__main__':
    main()
