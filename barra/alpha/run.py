"""
多信号Alpha预测 - 每日运行脚本
"""
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import qlib

from config import PROVIDER_URI
from barra.alpha.alpha_engine import AlphaEngine
from prefect import flow
from utils import LoggerFactory, is_trade_day
from utils.prefect import email_send_message_flow

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


@flow(name='barra_alpha', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(now_date: str = '',
         history_months: int = 24,
         market: str = 'csi300',
         portfolio: str = 'default',
         use_cache: bool = False):
    '''Prefect flow: 多信号Alpha每日预测'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return
    try:
        init_qlib()
        run(
            calc_date=now_date,
            history_months=history_months,
            market=market,
            output_dir=f'output/{now_date}',
            portfolio=portfolio,
            use_cache=use_cache,
        )
    except:
        err_msg = 'barra_alpha_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: barra_alpha', msg=err_msg)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多信号Alpha每日预测 — Prefect flow')
    parser.add_argument('--now-date', type=str, default='',
                        help='计算日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--history-months', type=int, default=24,
                        help='历史数据月数')
    parser.add_argument('--market', type=str, default='csi300',
                        help='市场代码，默认 csi300')
    parser.add_argument('--portfolio', type=str, default='default',
                        help='持仓组合名称，默认 default')
    parser.add_argument('--use-cache', action='store_true',
                        help='使用缓存数据')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="run.py:flow",
        ).deploy(
            name="barra_alpha",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            now_date=args.now_date,
            history_months=args.history_months,
            market=args.market,
            portfolio=args.portfolio,
            use_cache=args.use_cache,
        )
