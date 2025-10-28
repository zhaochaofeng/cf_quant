'''
    检查一段时间 MySQL 与 Tushare 数据是否一致
'''

import time
import fire
import traceback
from utils import send_email
from utils import CheckMySQLData

feas = ['ts_code', 'day', 'close', 'turnover_rate', 'turnover_rate_f',
        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
        'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
        'circ_mv']


def main(start_date: str, end_date: str):
    try:
        t = time.time()
        check = CheckMySQLData(
            table_name='valuation_ts',
            feas=feas,
            start_date=start_date,
            end_date=end_date,
            ts_api_func='daily_basic',
            log_file='logs/{}_check.log'.format(end_date)
        )

        df_mysql = check.fetch_data_from_mysql()
        stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()
        df_ts = check.fetch_data_from_ts(stocks)
        res = check.check(df_mysql, df_ts)
        if len(res) != 0:
            send_email('Data:Check:valuation_ts', '\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:valuation_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
