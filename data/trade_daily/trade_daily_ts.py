'''
    Tushare 日线行情数据入库MySQL — Prefect flow
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import time
import traceback
from datetime import datetime
from prefect import flow
from data.process_data import TSTradeDailyData
from utils import email_send_message_flow
from utils import is_trade_day

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'day': 'trade_date',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'pre_close': 'pre_close',
    'change': 'change',
    'pct_chg': 'pct_chg',
    'vol': 'vol',
    'amount': 'amount',
    'adj_factor': 'adj_factor'
}


@flow(name='trade_daily_ts', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(
    start_date: str = '',
    end_date: str = '',
    now_date: str = '',
    is_alive: bool = True,
    use_trade_day: bool = True
):
    '''Prefect flow: 每日定时拉取日线行情数据'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return

    try:
        t = time.time()
        processor = TSTradeDailyData(
            start_date, end_date, now_date,
            use_trade_day=use_trade_day,
            feas=feas,
            table_name='trade_daily_ts'
        )
        stocks = processor.get_stocks(is_alive=is_alive)
        df = processor.fetch_data_from_api(stocks, api_fun='daily', batch_size=1000, req_per_min=600)
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = 'trade_daily_ts_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: trade_daily_ts', msg=err_msg)
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
            entrypoint="trade_daily_ts.py:flow",
        ).deploy(
            name="trade_daily_ts",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            start_date=args.start_date,
            end_date=args.end_date,
            now_date=args.now_date
        )
