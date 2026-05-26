"""
投资组合优化每日运行入口
"""
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

import argparse
import traceback
from datetime import datetime

import qlib

from config import PROVIDER_URI
from portfolio_engine import PortfolioEngine
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


@flow(name='barra_portfolio', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 3)
def flow(now_date: str = '',
         value: float = 1e8,
         position: str = 'mysql',
         risk_aversion: float = 0.05,
         max_turnover: float = 1.0,
         use_qp: bool = False,
         save_mysql: bool = True,
         portfolio: str = 'default'):
    '''Prefect flow: 主动投资组合优化'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return

    print(f'计算日期: {now_date}')
    print(f'组合净值: {value:,.0f}元')
    print(f'风险厌恶系数: {risk_aversion}')
    print(f'换手率上限: {max_turnover}')

    try:
        init_qlib()
        engine = PortfolioEngine(
            calc_date=now_date,
            risk_output_dir=f'{project_root}/barra/risk_control/output/{now_date}',
            output_dir=f'{project_root}/barra/portfolio/output/{now_date}',
            portfolio_name=portfolio,
            risk_aversion=risk_aversion,
            max_turnover=max_turnover,
            position=position,
        )
        engine.run(
            portfolio_value=value,
            use_qp_init=use_qp,
            save_to_mysql=save_mysql,
        )
    except:
        err_msg = 'barra_portfolio_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: barra_portfolio', msg=err_msg)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='主动投资组合优化 — Prefect flow')
    parser.add_argument('--now-date', type=str, default='',
                        help='计算日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--value', type=float, default=1e8,
                        help='组合净值（元），默认1亿')
    parser.add_argument('--position', type=str, default='zero',
                        help='当前持仓读取方式。zero: 空仓; mysql: 从MySQL读取')
    parser.add_argument('--risk-aversion', type=float, default=0.05,
                        help='风险厌恶系数，默认0.05')
    parser.add_argument('--max-turnover', type=float, default=0.10,
                        help='换手率上限，默认0.10')
    parser.add_argument('--use-qp', action='store_true',
                        help='是否使用QP优化作为初始解')
    parser.add_argument('--save-mysql', action='store_true',
                        help='是否保存到MySQL')
    parser.add_argument('--portfolio', type=str, default='default',
                        help='组合名称')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="run.py:flow",
        ).deploy(
            name="barra_portfolio",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            now_date=args.now_date,
            value=args.value,
            position=args.position,
            risk_aversion=args.risk_aversion,
            max_turnover=args.max_turnover,
            use_qp=args.use_qp,
            save_mysql=args.save_mysql,
            portfolio=args.portfolio,
        )
