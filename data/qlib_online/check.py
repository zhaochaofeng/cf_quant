'''
    检查一段时间 MySQL 与 Tushare 数据是否一致
'''

import time
import fire
import traceback
from utils import send_email
from data.check_data import CheckMySQLData


feas = {
    'day': 'day',
    'qlib_code': 'qlib_code',
    'open': 'open',
    'close': 'close',
    'high': 'high',
    'low': 'low',
    'amount': 'amount',
    'adj_factor': 'factor'
}


def main(start_date: str,
         end_date: str,
         provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
         epsilon: float = 0.001,
         index_list=None
         ):
    if index_list is None:
        index_list = ['SH000300', 'SH000903', 'SH000905', 'SH000906']
    t = time.time()
    check = CheckMySQLData(
        start_date=start_date,
        end_date=end_date,
        feas=list(feas.values())
    )

    try:
        if not check.is_trade_day:
            return

        df_qlib, stocks = check.fetch_data_from_qlib(provider_uri=provider_uri, index_list=index_list)
        df_qlib.sort_index(inplace=True)
        stocks = ['{}.{}'.format(s[2:8], s[0:2]) for s in stocks]
        df_ts = check.fetch_data_from_ts(stocks, api_fun='pro_bar',
                                         batch_size=1, ts_type='ts',
                                         code_type='qlib', feas=list(feas.keys()),
                                         adj='hfq', adjfactor=True
                                         )
        df_ts.reset_index(inplace=True)
        df_ts.rename(columns=feas, inplace=True)
        df_ts.set_index(list(feas.values())[0:2], inplace=True)
        df_ts.sort_index(inplace=True)
        df_ts['amount'] = df_ts['amount'] * 1000

        res = check.check(df_qlib, df_ts, is_repair=False, compare_type='round', epsilon=epsilon)
        if len(res) != 0:
            send_email('Data:Check:qlib_online', '\n'.join(res))
        check.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        check.logger.error(error_msg)
        send_email('Data:Check:qlib_online', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
