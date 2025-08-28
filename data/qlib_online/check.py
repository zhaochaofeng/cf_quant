'''
    功能：数据质量检查
'''

import time
import qlib
from qlib.data import D
from datetime import datetime
import pandas as pd
from utils.utils import tushare_ts,tushare_pro
from utils.utils import send_email
from utils.utils import is_trade_day
import traceback
from utils.utils import mysql_connect

import warnings
warnings.filterwarnings('ignore')

is_update = False

def qlib_init():
    provider_uri = '~/.qlib/qlib_data/custom_data_hfq'
    qlib.init(provider_uri=provider_uri)

def get_qlib_data(start_date, end_date):
    ''' qlib线上数据 '''
    print('get_qlib_data ...')
    fields = ['$open', '$close', '$high', '$low']
    instruments = ['SZ300760']
    # instruments_config = D.instruments(market='all')
    # instruments = D.list_instruments(instruments=instruments_config, start_time=start_date, end_time=end_date, as_list=True)
    # index_list = ['SH000300', 'SH000903', 'SH000905']
    # instruments = list(set(instruments) - set(index_list))
    # instruments = instruments[0:100]
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
        time.sleep(0.075)  # 每分钟只能访问800次。60/800=0.075
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

def get_and_update_factor(code, date, conn, pro):
    ts_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    try:
        with conn.cursor() as cursor:
            sql = '''
                select adj_factor from trade_daily2 where day=%s and ts_code=%s
            '''
            # print(cursor.mogrify(sql, (date, code)))
            cursor.execute(sql, (date, code))
            mysql_factor = float(cursor.fetchone()[0])
    except Exception as e:
        raise Exception('mysql factor request fail: '.format(e))

    try:
        ts_factor = pro.adj_factor(ts_code=code, trade_date=ts_date)
        ts_factor = ts_factor.iloc[0]['adj_factor']
    except Exception as e:
        raise Exception('tushare factor request fail: '.format(e))

    if mysql_factor != ts_factor:
        global is_update
        is_update = True
        print('{} {} factor is not equal. mysql_factor: {}, ts_factor: {}'.format(code, date, mysql_factor, ts_factor))
        try:
            with conn.cursor() as cursor:
                sql = '''
                    update trade_daily2 set adj_factor=%s where day=%s and ts_code=%s
                '''
                print(cursor.mogrify(sql, (ts_factor, date, code)))
                cursor.execute(sql, (ts_factor, date, code))
                conn.commit()
                print('update trade_daily2 adj_factor completed !')
        except Exception as e:
            conn.rollback()
            raise Exception('mysql update factor fail: '.format(e))

    return mysql_factor, ts_factor

def format_email_info(qlib_df, ts_df):
    diff = qlib_df - ts_df
    mask_na = diff.isna().any(axis=1)
    mask_gt = (abs(diff) > 0.1).any(axis=1)

    index_gt = diff.index[mask_gt]
    index_na = diff.index[mask_na]

    res = []
    # 值不相等的情况
    field = ['open', 'close', 'high', 'low']
    conn = mysql_connect()
    pro = tushare_pro()
    for index in index_gt:
        qlib_f = []
        ts_f = []
        code = '{}.{}'.format(index[0][2:8], index[0][0:2])
        date = index[1].strftime('%Y-%m-%d')
        mysql_factor, ts_factor = get_and_update_factor(code, date, conn, pro)
        qlib_f.append('factor:{}'.format(mysql_factor))
        ts_f.append('factor: {}'.format(ts_factor))

        qlib_row = qlib_df.loc[index]
        ts_row = ts_df.loc[index]

        for f in field:
            qlib_f.append('{}:{}'.format(f, str(round(qlib_row[f], 4))))
            ts_f.append('{}:{}'.format(f, str(round(ts_row[f], 4))))

        res.append('{}: [qlib: {} | ts: {}]'.format(
            code + "_" + date,
            ', '.join(qlib_f),
            ', '.join(ts_f)
        ))

    # qlib 数据有缺失情况
    for index in index_na:
        res.append('{}: qlib is NaN'.format(index[0] + "_" + index[1].strftime('%Y-%m-%d')))
    return res

def main():
    try:
        qlib_init()

        # start_date = '2015-01-05'
        # end_date = '2025-08-08'
        start_date = '2025-08-10'
        end_date = datetime.now().strftime('%Y-%m-%d')
        if not is_trade_day(end_date):
            print('非交易日，不检查！')
            exit(0)

        print('start_date: {}, end_date: {}'.format(start_date, end_date))
        qlib_df, instruments = get_qlib_data(start_date, end_date)
        print('instruments len: {}'.format(len(instruments)))
        ts_df = get_tushare_data(instruments, start_date, end_date)
        print('qlib_df shape: {}, ts_df shape: {}'.format(qlib_df.shape, ts_df.shape))
        print(qlib_df.head())
        print('-' * 100)
        print(ts_df.head())

        res = format_email_info(qlib_df, ts_df)
        if len(res) > 0:
            send_email("Data: qlib_online", '\n'.join(res))
    except Exception as e:
        print(e)
        error_info = traceback.print_exc()
        send_email("Data: qlib_online", error_info)

if __name__ == '__main__':
    t = time.time()
    main()
    print('耗时：{}s'.format(round(time.time()-t, 4)))
