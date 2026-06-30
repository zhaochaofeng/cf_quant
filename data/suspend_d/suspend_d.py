'''
    Tushare 停复盘数据 — Prefect flow
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import time
import traceback
from datetime import datetime
from prefect import flow
from data.process_data import SuspendD
from utils import email_send_message_flow
from utils import is_trade_day

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'day': 'trade_date',
    'suspend_timing': 'suspend_timing',
    'suspend_type': 'suspend_type'
}


@flow(name='suspend_d', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(
    start_date: str = '',
    end_date: str = '',
):
    '''Prefect flow: 每日定时拉取日线行情数据'''
    now_date = datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return

    try:
        t = time.time()
        processor = SuspendD(
            start_date, end_date,
            feas=feas,
            table_name='suspend_d',
        )
        df = processor.fetch_data_from_api()
        if df.empty:
            print('df is empty. start_date: {}, end_date: {}'.format(start_date, end_date))
            return

        data = processor.process(df)
        processor.write_to_mysql(data)
        print('data len: {}'.format(len(data)))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = 'suspend_d_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: suspend_d', msg=err_msg)
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="suspend_d.py:flow",
        ).deploy(
            name="suspend_d",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            start_date=args.start_date,
            end_date=args.end_date
        )
