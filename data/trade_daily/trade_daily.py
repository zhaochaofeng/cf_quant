'''
    功能：tushare表导入mysql
    描述：日级交易数据，取数周期为365天，每日覆盖更新
'''

import time
import argparse
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import tushare_pro, is_trade_day
import warnings
warnings.filterwarnings("ignore")

config = get_config()
pro = tushare_pro()

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

def main(args):
    # tushare数据
    ts_codes = pro.stock_basic()['ts_code'].values.tolist()
    print('-' * 100)
    print('股票数：{}'.format(len(ts_codes)))
    conn = mysql_connect()
    # 先清空表
    with conn.cursor() as cursor:
        cursor.execute('delete from trade_daily;')
        conn.commit()
        print('清空表成功 !!!')

    start_date = (datetime.strptime(args.date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y%m%d')
    end_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y%m%d')
    print('start_date: {}, end_date: {}'.format(start_date, end_date))

    with conn.cursor() as cursor:
        fea_format = ['%({})s'.format(f) for f in list(fea_to_from.keys())]
        fea_keys = ','.join(list(fea_to_from.keys()))
        fea_keys = fea_keys.replace('change', '`change`')

        sql = '''
                INSERT INTO trade_daily({}) VALUES ({})
            '''.format(fea_keys, ','.join(fea_format))

        for i, code in enumerate(ts_codes):
            # time.sleep(0.1)  # API调用频次：1min不超过700次
            if (i+1) % 100 == 0:
                print('process: {}'.format(i+1))

            data = []
            info = ts.pro_bar(ts_code=code, start_date=start_date,
                              end_date=end_date, asset='E',
                              adj='qfq', freq='D')
            if info.empty:
                continue
            for index, row in info.iterrows():
                try:
                    tmp = parse_line(row, fea_to_from)
                    data.append(tmp)
                except Exception as e:
                    print('except: {}'.format(e))
                    print('code: {}'.format(code))
                    continue
            try:
                if i == 0:
                    s = cursor.mogrify(sql, data[0])
                    print(s)
                cursor.executemany(sql, data)
                conn.commit()
            except Exception as e:
                print('except: {}'.format(e))
                conn.rollback()  # 回滚
                exit(1)
    conn.close()
    print('写入完成!!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-07-04')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('耗时：{}s'.format(round(time.time() - t, 4)))


