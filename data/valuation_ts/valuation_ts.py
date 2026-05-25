'''
    Tushare 市值数据入库MySQL — Prefect flow
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import time
import traceback
from datetime import datetime
from prefect import flow
from data.process_data import TSCommonData
from utils import is_trade_day
from utils.prefect import email_send_message_flow

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'day': 'trade_date',
    'close': 'close',
    'turnover_rate': 'turnover_rate',
    'turnover_rate_f': 'turnover_rate_f',
    'volume_ratio': 'volume_ratio',
    'pe': 'pe',
    'pe_ttm': 'pe_ttm',
    'pb': 'pb',
    'ps': 'ps',
    'ps_ttm': 'ps_ttm',
    'dv_ratio': 'dv_ratio',
    'dv_ttm': 'dv_ttm',
    'total_share': 'total_share',
    'float_share': 'float_share',
    'free_share': 'free_share',
    'total_mv': 'total_mv',
    'circ_mv': 'circ_mv'
}


@flow(name='valuation_ts', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(start_date: str = None, end_date: str = None, now_date: str = ''):
    '''Prefect flow: 每日定时拉取市值数据'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return

    try:
        t = time.time()
        processor = TSCommonData(
            start_date,
            end_date,
            now_date,
            use_trade_day=True,
            feas=feas,
            table_name='valuation_ts'
        )

        stocks = processor.get_stocks(is_alive=True)
        df = processor.fetch_data_from_api(stocks, api_fun='daily_basic', batch_size=1, req_per_min=700)
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = 'valuation_ts_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: valuation_ts', msg=err_msg)
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--now-date', type=str, default='',
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="valuation_ts.py:flow",
        ).deploy(
            name="valuation_ts",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            start_date=args.start_date,
            end_date=args.end_date,
            now_date=args.now_date,
        )
