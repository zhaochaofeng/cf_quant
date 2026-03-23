"""
Barra CNE6 风险模型 - 每月运行脚本
用于更新模型参数（因子收益率历史、协方差矩阵、特异风险）
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qlib
from barra.risk_control.barra_engine import BarraRiskEngine
from utils import dt, LoggerFactory, send_email
import traceback

logger = LoggerFactory.get_logger(__name__)

def init_qlib():
    """初始化 qlib，注册 PTTM 自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri='~/.qlib/qlib_data/custom_data_hfq',
        custom_ops=[PTTM]
    )
    logger.info("Qlib 初始化完成（已注册 PTTM 操作符）")


def run_monthly_update(end_date: str, history_months: int = 120,
                       output_dir: str = 'output',
                       n_jobs: int = 4,
                       use_cache: bool = True
                       ):
    """
    运行月度模型更新
    
    Args:
        end_date: 数据截止日期
        history_months: 历史数据月数，默认120（10年）
        output_dir: 输出目录，默认为 'output'
        n_jobs: 并行计算核心数，默认4
    """
    # 计算开始日期（准确计算月份，考虑不同月份的天数差异）
    start_date = dt.subtract_months(end_date, history_months)

    logger.info(f"{'='*70}")
    logger.info(f"Barra CNE6 月度模型更新")
    logger.info(f"数据区间：{start_date} 至 {end_date}")
    logger.info(f"{'='*70}")

    # 初始化引擎
    engine = BarraRiskEngine(
        calc_date=end_date,
        output_dir=output_dir,
        n_jobs=n_jobs
    )
    
    # 运行月频更新
    engine.run_monthly_update(start_date, end_date, use_cache)

    '''
    # 保存模型数据到文件，供日频计算使用
    logger.info("\n保存模型数据...")
    saved_files = engine.save_model_data('barra/risk_control/model_data')
    logger.info(f"模型数据已保存，共 {len(saved_files)} 个文件")
    
    logger.info(f"\n{'='*70}")
    logger.info("月度模型更新完成")
    logger.info(f"{'='*70}")
    '''

def main():
    try:
        parser = argparse.ArgumentParser(description='Barra CNE6 月度模型更新')
        parser.add_argument('--end-date', type=str, required=True,
                           help='数据截止日期，格式YYYY-MM-DD')
        parser.add_argument('--history-months', type=int, default=72,
                           help='历史数据月数，默认72(1+5年)')
        parser.add_argument('--output_dir', type=str, default='output',
                           help='输出路径')
        parser.add_argument('--n-jobs', type=int, default=4,
                           help='并行计算核心数，默认4')
        parser.add_argument('--use-cache', action='store_true', help='是否使用缓存，用于实验阶段')

        args = parser.parse_args()
        print(args)

        # 初始化qlib
        init_qlib()

        # 运行月度更新
        run_monthly_update(
            end_date=args.end_date,
            history_months=args.history_months,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs,
            use_cache=args.use_cache
        )
    except Exception as e:
        logger.error(f"运行月度更新时出错：{e}")
        send_email(f"Barra CNE6 模型更新出错", traceback.format_exc())
        raise Exception('Barra CNE6 模型更新出错')


if __name__ == '__main__':
    main()
