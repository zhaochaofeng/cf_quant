'''
    功能：由于复权因子在除权除息日附近可能变动，每天检查历史复权因子，如果发生变动，则更新历史数据
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from datetime import datetime

import time
import traceback
from utils import email_send_message_flow, get_n_pretrade_day, is_trade_day
from data.check_data import CheckMySQLData
from prefect import flow
import argparse
from pathlib import Path

feas = ['day', 'ts_code', 'adj_factor']

@flow(name='update_factor', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(now_date: str = '', interval: int = 30, use_trade_day: bool = True):
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    end_date = now_date
    start_date = get_n_pretrade_day(end_date, interval)
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='trade_daily_ts',
            feas=feas,
            use_trade_day=use_trade_day
        )

        df_mysql = check.fetch_data_from_mysql()
        stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()
        df_ts = check.fetch_data_from_ts(stocks, api_fun='adj_factor')
        df_ts = df_ts.reindex(df_mysql.index)  # 对齐索引
        res = check.check(df_mysql, df_ts, is_repair=True)
        if len(res) != 0:
            email_send_message_flow(subject='Data:update_factor update factor(Auto Repair)', msg='\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        err_msg = 'update_factor_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: update_factor', msg=err_msg)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, help='当前日期', default=None)
    parser.add_argument('--interval', type=int, default=30, help='间隔天数')
    parser.add_argument('--deploy', action='store_true', help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="update_factor.py:flow"
        ).deploy(
            name="update_factor",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            now_date=args.now_date,
            interval=args.interval
        )

