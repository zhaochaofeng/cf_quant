'''
    功能：数据质量检查
'''

import time
import qlib
from qlib.data import D
from datetime import datetime
import pandas as pd
from utils.utils import tushare_ts
from utils.utils import send_email
from utils.utils import is_trade_day
import traceback

import warnings
warnings.filterwarnings('ignore')

def qlib_init():
    provider_uri = '~/.qlib/qlib_data/custom_data_hfq'
    qlib.init(provider_uri=provider_uri)

def get_qlib_data(start_date, end_date):
    ''' qlib线上数据 '''
    print('get_qlib_data ...')
    fields = ['$open', '$close', '$high', '$low']
    # instruments = ['SH600530', 'SH600535']
    instruments_config = D.instruments(market='all')
    instruments = D.list_instruments(instruments=instruments_config, start_time=start_date, end_time=end_date, as_list=True)
    index_list = ['SH000300', 'SH000903', 'SH000905']
    instruments = list(set(instruments) - set(index_list))
    instruments = instruments[0:500]
    qlib_df = D.features(instruments, fields, start_time=start_date, end_time=end_date)
    qlib_df.columns = ['open', 'close', 'high', 'low']
    qlib_df.sort_index(inplace=True)
    na = qlib_df.isna().any(axis=1)
    qlib_df = qlib_df[~na]
    return qlib_df,  instruments

def get_tushare_data(instruments, start_date, end_date):
    ''' tushare API 数据'''
    print('get_tushare_data ...')
    ts = tushare_ts()
    df = pd.DataFrame()
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    for i, instrument in enumerate(instruments):
        if (i+1) % 100 == 0:
            print('process: {}'.format(i+1))
        instrument = '{}.{}'.format(instrument[2:8], instrument[0:2])
        tmp_df = ts.pro_bar(ts_code=instrument, start_date=start_date, end_date=end_date, adj='hfq')
        tmp_df = tmp_df[['ts_code', 'trade_date', 'open', 'close', 'high', 'low']]
        tmp_df.columns = ['instrument', 'datetime', 'open', 'close', 'high', 'low']
        tmp_df.loc[:, 'instrument'] = tmp_df['instrument'].apply(lambda x: '{}{}'.format(x[7:9], x[0:6]))
        df = pd.concat([df, tmp_df], axis=0, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d')
    df.set_index(keys=['instrument', 'datetime'], inplace=True)
    df.sort_index(inplace=True)
    na = df.isna().any(axis=1)
    df = df[~na]
    return df

def main():
    try:
        qlib_init()

        start_date = '2015-01-05'
        # end_date = '2025-08-08'
        end_date = datetime.now().strftime('%Y-%m-%d')
        if not is_trade_day(end_date):
            print('非交易日，不检查！')
            exit(0)

        # end_date = '2025-08-22'
        print('start_date: {}, end_date: {}'.format(start_date, end_date))
        qlib_df, instruments = get_qlib_data(start_date, end_date)
        print('instruments len: {}'.format(len(instruments)))
        ts_df = get_tushare_data(instruments, start_date, end_date)
        print('qlib_df shape: {}, ts_df shape: {}'.format(qlib_df.shape, ts_df.shape))
        print(qlib_df.head())
        print('-' * 100)
        print(ts_df.head())

        diff = qlib_df - ts_df
        mask = ((abs(diff) > 0.1) | diff.isna()).any(axis=1)
        result_index = diff.index[mask]
        res = []
        for item in result_index.values:
            res.append('{}:{}'.format(item[0], item[1].strftime('%Y-%m-%d')))
        if result_index.shape[0] > 0:
            send_email("Data: qlib_online", '\n'.join(res))
    except Exception as e:
        print(e)
        error_info = traceback.print_exc()
        send_email("Data: qlib_online", error_info)

if __name__ == '__main__':
    t = time.time()
    main()
    print('耗时：{}s'.format(round(time.time()-t, 4)))
