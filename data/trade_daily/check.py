'''
    功能：检查trade_daily_ts 与 trade_daily_bao数据是否一致
'''

import time
import fire
import traceback
from datetime import datetime
from utils import send_email
from data.check_data import CheckMySQLData


feas = ['day', 'qlib_code', 'open', 'close', 'high', 'low', 'vol', 'amount']


def main(start_date: str,
         end_date: str,
         now_date: str = None):
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='trade_daily_ts',
            feas=feas
        )
        now_date = now_date if now_date else datetime.now().strftime('%Y-%m-%d')

        sql_ts = '''
                select {} from
                (select qlib_code as qlib from stock_info_ts where exchange in ('SSE', 'SZSE') and day='{}')a
                JOIN
                (select {} from trade_daily_ts where day>='{}' and day<='{}')b
                ON
                a.qlib = b.qlib_code;
            '''.format(','.join(feas), now_date, ','.join(feas), start_date, end_date)

        sql_bao = '''
                select {} from
                (select qlib_code as qlib from stock_info_bao where exchange in ('SSE', 'SZSE') and day='{}')a
                JOIN
                (select {} from trade_daily_bao where day>='{}' and day<='{}')b
                ON
                a.qlib = b.qlib_code;
            '''.format(','.join(feas), now_date, ','.join(feas), start_date, end_date)

        df_ts = check.fetch_data_from_mysql(sql_str=sql_ts)
        df_bao = check.fetch_data_from_mysql(sql_str=sql_bao)
        res = check.check(df_ts, df_bao, is_repair=False)
        if len(res) != 0:
            send_email('Data:Check:trade_daily_ts', '\n'.join(res))
        check.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:trade_daily_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
