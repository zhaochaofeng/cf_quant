'''
    检查一段时间 MySQL 与 Tushare 数据是否一致 — Prefect flow
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import time
import traceback
from datetime import datetime, timedelta
from prefect import flow
from data.check_data import CheckMySQLData
from utils.prefect import email_send_message_flow

feas = ['day', 'ts_code', 'close', 'turnover_rate', 'turnover_rate_f',
        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
        'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
        'circ_mv']


@flow(name='check_valuation_ts', log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(now_date: str = None, use_trade_day: bool = True):
    '''Prefect flow: 检查 MySQL 与 Tushare 数据是否一致'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.strptime(now_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = now_date

    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='valuation_ts',
            feas=feas,
            use_trade_day=use_trade_day,
        )

        if not check.is_trade_day:
            print(f'{end_date} 非交易日，跳过')
            return

        df_mysql = check.fetch_data_from_mysql()
        stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()
        df_ts = check.fetch_data_from_ts(stocks, api_fun='daily_basic', batch_size=1, req_per_min=700)
        res = check.check(df_mysql, df_ts, is_repair=True)
        if len(res) != 0:
            msg = '\n'.join(res)
            print(msg)
            email_send_message_flow(subject='Data:Check:valuation_ts(Auto Repair)', msg=msg)
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = 'check_valuation_ts({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data:Check:valuation_ts', msg=err_msg)
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, default=None,
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--no-trade-day', action='store_false', dest='use_trade_day',
                        help='不限制交易日参数')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="check.py:flow",
        ).deploy(
            name="check_valuation_ts",
            work_pool_name="cf_quant",
        )
    else:
        flow(
            now_date=args.now_date,
            use_trade_day=args.use_trade_day,
        )
