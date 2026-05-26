'''
    功能：由于复权因子在除权除息日附近可能变动，每天检查历史复权因子，如果发生变动，则更新历史数据
'''

import time
import fire
import traceback
from utils import send_email
from data.check_data import CheckMySQLData


feas = ['day', 'ts_code', 'adj_factor']


def main(start_date: str, end_date: str, use_trade_day: bool = True):
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
            send_email('Data:Check:trade_daily_ts update factor(Auto Repair)', '\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:trade_daily_ts:update factor', error_msg)


if __name__ == '__main__':
    fire.Fire(main)


