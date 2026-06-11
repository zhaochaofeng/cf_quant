"""
    因子暴露度计算
"""
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from prefect import flow
from prefect.logging import get_run_logger

from utils import init_qlib
from utils import dt, get_trade_cal_inter, is_trade_day, email_send_message_flow
from barra.factors.data_loader import DataLoader
from barra.factors.exposure import CNE6IndExposure


def run(calc_date: str,
        history_months: int = 24,
        output: str = 'data',
        n_jobs: int = 4,
        extend_start: int = 6
        ):
    start_date = dt.subtract_months(calc_date, history_months)
    end_date = calc_date
    print(f'start_date: {start_date}, calc_date: {end_date}')
    data_loader = DataLoader()
    exp_builder = CNE6IndExposure()
    instruments = data_loader.load_instruments(start_date, end_date)
    industry_df = data_loader.load_industry(instruments, start_date, end_date)
    market_cap_df = data_loader.load_market_cap(instruments, start_date, end_date)
    raw_data = data_loader.load_fields_data(instruments, start_date, calc_date,
                                            extend_start=extend_start, extend_freq='Y')
    com_date = get_trade_cal_inter(start_date, end_date)
    com_date = pd.to_datetime(com_date)
    exp_builder.build_exposure_matrix(raw_data, industry_df, market_cap_df, n_jobs, com_date, output)


@flow(name='factors_exposure', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(now_date: str=''):
    logger = get_run_logger()
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        logger.warning(f'{now_date} 非交易日，跳过')
        return

    try:
        init_qlib()
        run(
            calc_date=now_date,
            history_months=args.history_months,
            output=f'data/{now_date}',
            n_jobs=args.n_jobs,
            extend_start=args.extend_start
        )
    except Exception as e:
        err_msg = 'factors_exposure_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        logger.error(err_msg)
        email_send_message_flow(subject='Data: barra_risk', msg=err_msg)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Barra 因子暴露度计算')
    parser.add_argument('--now-date', type=str, default='',
                        help='计算日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--history-months', type=int, default=36,
                        help='历史数据月数')
    parser.add_argument('--n-jobs', type=int, default=os.cpu_count() - 2,
                        help='并行计算核心数')
    parser.add_argument('--extend_start', type=int, default=6, help='扩展数据起始年数')
    parser.add_argument('--deploy', action='store_true', help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint='run.py:flow'
        ).deploy(
            name='factors_exposure',
            work_pool_name='cf_quant'
        )
    else:
        init_qlib()
        run(
            calc_date=args.now_date,
            history_months=args.history_months,
            output=f'data/{args.now_date}',
            n_jobs=args.n_jobs,
            extend_start=args.extend_start,
        )
