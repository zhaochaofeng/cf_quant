import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from data.process_data import ShiborRate
from datetime import datetime
from prefect import flow
from utils import is_trade_day, email_send_message_flow
import traceback
import argparse


feas = {
     'date': 'date',
     'on_rate': 'on',
     '1w': '1w',
     '2w': '2w',
     '1m': '1m',
     '3m': '3m',
     '6m': '6m',
     '9m': '9m',
     '1y': '1y'
}

@flow(name='shibor_rate', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(start_date: str = '', end_date: str = ''):
    now_date = datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date

    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return
    try:
        processor = ShiborRate(
            start_date=start_date,
            end_date=end_date,
            feas=feas,
            table_name='shibor',
        )
        df = processor.fetch_data_from_api()
        data = processor.process(df)
        processor.write_to_mysql(data)
    except Exception as e:
        err_msg = 'shibor_rate_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: trade_daily_ts', msg=err_msg)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, help='start date')
    parser.add_argument('--end-date', type=str, help='end date')
    parser.add_argument('--deploy', action='store_true', help='deploy')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint='shibor.py:flow'
        ).deploy(
            name='shibor_rate',
            work_pool_name='cf_quant'
        )
    else:
        flow(args.start_date, args.end_date)
