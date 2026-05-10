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

from barra.alpha.config import QLIB_PROVIDER_URI, OUTPUT_DIR
from barra.alpha.alpha_engine import AlphaEngine
from utils import LoggerFactory, send_email

logger = LoggerFactory.get_logger(__name__)


def init_qlib():
    """初始化qlib，注册PTTM自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri=QLIB_PROVIDER_URI,
        custom_ops=[PTTM]
    )
    logger.info('Qlib初始化完成')


def run_alpha(calc_date: str, market: str = 'csi300',
              output_dir: str = OUTPUT_DIR,
              portfolio: str = 'default') -> None:
    """运行每日Alpha预测

    Args:
        calc_date: 计算日期
        market: 市场代码
        output_dir: 输出目录
        portfolio: 持仓组合名称
    """
    engine = AlphaEngine(market=market, output_dir=output_dir)
    engine.run(calc_date, portfolio=portfolio)


def main():
    try:
        parser = argparse.ArgumentParser(description='多信号Alpha每日预测')
        parser.add_argument('--calc_date', type=str, required=True,
                            help='计算日期，如 2026-03-06')
        parser.add_argument('--market', type=str, default='csi300',
                            help='市场代码，默认 csi300')
        parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                            help='输出目录')
        parser.add_argument('--portfolio', type=str, default='default',
                            help='持仓组合名称，默认 default')
        args = parser.parse_args()

        init_qlib()
        run_alpha(args.calc_date, args.market, args.output_dir, args.portfolio)
    except Exception as e:
        logger.error(f'运行出错: {e}')
        send_email(f'Alpha预测每日计算出错: {e}', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
