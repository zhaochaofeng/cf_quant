'''
    功能：检查stock_info_ts 与 stock_info_bao数据是否一致
'''

import time
import fire
import traceback
from utils import send_email
from data.check_data import CheckMySQLData


feas = ['day', 'qlib_code', 'list_date', 'status', 'exchange']

def main(now_date: str):
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=now_date,
            end_date=now_date,
            table_name='stock_info_ts',
            feas=feas
        )

        conditions_dict = {
            'list_date <=': now_date,
            'exchange IN': ('SSE', 'SZSE')
        }

        df_ts = check.fetch_data_from_mysql(table_name='stock_info_ts', conditions_dict=conditions_dict)
        df_bao = check.fetch_data_from_mysql(table_name='stock_info_bao', conditions_dict=conditions_dict)
        res = check.check(df_ts, df_bao, is_repair=False)
        if len(res) != 0:
            send_email('Data:Check:stock_info_ts', '\n'.join(res))
        check.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:stock_info_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
