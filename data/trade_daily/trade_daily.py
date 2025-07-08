'''
    功能：tushare表导入mysql
    描述：日级交易数据，取数周期为365天，每日覆盖更新
'''

import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import tushare_ts, tushare_pro, is_trade_day
import warnings
warnings.filterwarnings("ignore")

config = get_config()
ts = tushare_ts()
pro = tushare_pro()
batch_size = 10000    # 每次写入mysql的行数

fea_to_from = {
     'ts_code': 'ts_code',
     'day': 'trade_date',
     'open': 'open',
     'high': 'high',
     'low': 'low',
     'close': 'close',
     'pre_close': 'pre_close',
     'change': 'change',
     'pct_chg': 'pct_chg',
     'vol': 'vol',
     'amount': 'amount'
}

def parse_line(row, fea_to_from):
    ''' 解析数据 '''
    tmp = {}
    for f in fea_to_from.keys():
        try:
            v = row[fea_to_from[f]]
            if f == 'day':
                # 日期格式转换
                v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
            if pd.isna(v):
                v = None
            tmp[f] = v
        except Exception as e:
            print('except: {}'.format(e))
    return tmp

def request_from_tushare(ts_codes):
    # 从tushare API获取数据
    print('-' * 100)
    print('从tushare API获取数据...')
    start_date = (datetime.strptime(args.date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y%m%d')
    end_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y%m%d')
    print('start_date: {}, end_date: {}'.format(start_date, end_date))
    data = []
    for i, code in enumerate(ts_codes):
        # time.sleep(0.1)  # API调用频次：1min不超过700次
        if (i + 1) % 100 == 0:
            print('requested num: {}'.format(i + 1))
        info = ts.pro_bar(ts_code=code, start_date=start_date,
                          end_date=end_date, asset='E',
                          adj='qfq', freq='D')
        if info is None:
            continue
        for index, row in info.iterrows():
            try:
                tmp = parse_line(row, fea_to_from)
                data.append(tmp)
            except Exception as e:
                print('except: {}'.format(e))
                print('code: {}'.format(code))
                continue
    return data
def write_to_mysql(data):
    print('-' * 100)
    print('导入msyql ...')
    conn = mysql_connect()
    # 先清空表
    with conn.cursor() as cursor:
        cursor.execute('delete from trade_daily;')
        conn.commit()
        print('清空表成功 !!!')

    with conn.cursor() as cursor:
        fea_format = ['%({})s'.format(f) for f in list(fea_to_from.keys())]
        fea_keys = ','.join(list(fea_to_from.keys()))
        fea_keys = fea_keys.replace('change', '`change`')

        sql = '''
                    INSERT INTO trade_daily({}) VALUES ({})
                '''.format(fea_keys, ','.join(fea_format))

        n = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
        print('n: {}'.format(n))
        for i in range(int(n)):
            try:
                if i == 0:
                    print(cursor.mogrify(sql, data[0]))
                cursor.executemany(sql, data[i * batch_size:min((i + 1) * batch_size, len(data))])
                conn.commit()
            except Exception as e:
                print('except: {}'.format(e))
                conn.rollback()
                exit(1)
    conn.close()
    print('写入完成!!!')

def main(args):
    t = time.time()
    # 1、股票集合
    ts_codes = pro.stock_basic()['ts_code'].values.tolist()
    print('-' * 100)
    print('股票数：{}'.format(len(ts_codes)))

    # 2、获取交易数据
    data = request_from_tushare(ts_codes)
    print('数据请求耗时：{}s'.format(round(time.time()-t, 4)))
    t = time.time()
    print('data len: {}'.format(len(data)))
    if len(data) < 100:
        print('tushare 数据请求失败 ！！！')
        exit(1)

    # 3、写入mysql
    write_to_mysql(data)
    print('数据写入耗时：{}s'.format(round(time.time() - t, 4)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-07-07')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('总耗时：{}s'.format(round(time.time() - t, 4)))
