'''
    检查一段时间 MySQL 与 Tushare 数据是否一致
'''

import time
import fire
import traceback
from utils import send_email
from data.check_data import CheckMySQLData


feas = ['ts_code', 'day', 'close', 'turnover_rate', 'turnover_rate_f',
        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
        'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
        'circ_mv']


def main(start_date: str, end_date: str, use_trade_day: bool = False):
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='valuation_ts',
            feas=feas,
            use_trade_day=use_trade_day
        )

        df_mysql = check.fetch_data_from_mysql()
        stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()
        df_ts = check.fetch_data_from_ts(stocks, api_fun='daily_basic', batch_size=1, req_per_min=700)
        res = check.check(df_mysql, df_ts, is_repair=True)
        if len(res) != 0:
            send_email('Data:Check:valuation_ts(Auto Repair)', '\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:valuation_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
