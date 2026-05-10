"""
多信号Alpha预测 - 每日运行脚本
"""
import sys
import argparse
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qlib

from config import PROVIDER_URI
from barra.alpha.alpha_engine import AlphaEngine
from utils import LoggerFactory, send_email

logger = LoggerFactory.get_logger(__name__)


def init_qlib():
    """初始化qlib，注册PTTM自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri=PROVIDER_URI,
        custom_ops=[PTTM]
    )
    logger.info('Qlib初始化完成 !')


def run(calc_date: str,
        history_months: int = 24,
        market: str = 'csi300',
        output_dir: str = 'output',
        portfolio: str = 'default',
        use_cache: bool = False) -> None:
    """运行每日Alpha预测

    Args:
        calc_date: 计算日期
        market: 市场代码
        output_dir: 输出目录
        portfolio: 持仓组合名称
        use_cache: 是否使用缓存数据
    """
    engine = AlphaEngine(market=market, output_dir=output_dir)
    engine.run(calc_date, history_months, portfolio=portfolio, use_cache=use_cache)


def main():
    try:
        parser = argparse.ArgumentParser(description='多信号Alpha每日预测')
        parser.add_argument('--calc_date', type=str, required=True,
                            help='计算日期，如 2026-04-24')
        parser.add_argument('--history-months', type=int, default=24,
                            help='历史数据月数')
        parser.add_argument('--market', type=str, default='csi300',
                            help='市场代码，默认 csi300')
        parser.add_argument('--output_dir', type=str, default='output',
                            help='输出目录，默认 output')
        parser.add_argument('--portfolio', type=str, default='default',
                            help='持仓组合名称，默认 default')
        parser.add_argument('--use-cache', action='store_true',
                            help='使用缓存数据（从output/debug/加载parquet）')
        args = parser.parse_args()

        init_qlib()
        run(args.calc_date,
            args.history_months,
            args.market,
            args.output_dir + f'/{args.calc_date}',
            args.portfolio,
            use_cache=args.use_cache)

    except Exception as e:
        logger.error(f'运行出错: {e}')
        send_email(f'Alpha预测每日计算出错: {e}', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
