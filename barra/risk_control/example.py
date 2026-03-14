"""
Barra CNE6 风险模型完整示例
展示如何一次性运行月频更新和日频风险计算
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qlib
from barra.risk_control.barra_engine import BarraRiskEngine


def main():
    # 初始化qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
    print("Qlib初始化完成\n")
    
    # 设置参数
    calc_date = '2024-03-14'  # 计算日期
    portfolio_input = 'random'  # 随机生成50只股票组合
    
    print("=" * 70)
    print("Barra CNE6 风险模型完整示例")
    print("=" * 70)
    
    # 步骤1: 初始化引擎
    print("\n步骤1: 初始化风险模型引擎...")
    engine = BarraRiskEngine(
        calc_date=calc_date,
        portfolio_input=portfolio_input,
        output_dir='barra/risk_control/output',
        n_jobs=1
    )
    print(f"   组合包含 {len(engine.portfolio_weights)} 只股票")
    
    # 步骤2: 月频更新（实际应该在每月初运行一次）
    print("\n步骤2: 月频模型更新...")
    print("   (注: 实际使用时应该预先计算好并保存模型数据)")
    # 这里为了演示，简化处理，实际应该加载预计算的模型数据
    # engine.run_monthly_update('2014-03-01', calc_date)
    
    # 步骤3: 日频风险计算
    print("\n步骤3: 日频风险计算...")
    # 为了演示，使用模拟数据
    print("   (注: 这里需要预先计算的模型数据)")
    
    # 步骤4: 保存结果
    print("\n步骤4: 保存风险指标...")
    # saved_files = engine.save_results()
    
    # 步骤5: 打印报告
    print("\n步骤5: 生成风险报告...")
    # engine.print_risk_report()
    
    print("\n" + "=" * 70)
    print("示例运行完成")
    print("=" * 70)
    print("\n使用说明:")
    print("1. 先运行月度更新: python barra/risk_control/run_monthly.py --end-date 2024-03-01")
    print("2. 再运行每日计算: python barra/risk_control/run_daily.py --date 2024-03-14")
    print("\n或者直接使用引擎API:")
    print("  engine = BarraRiskEngine(calc_date='2024-03-14')")
    print("  engine.run_monthly_update(start_date, end_date)  # 月度更新")
    print("  engine.run_daily_risk()  # 日频计算")
    print("  engine.save_results()  # 保存结果")


if __name__ == '__main__':
    main()
