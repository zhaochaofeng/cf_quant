'''
    功能：股票基本信息表（沪深京3个交易所股票）
    数据来源：Tushare
    表：stock_info
'''

import time
import argparse
import pandas as pd
import tushare as ts
from datetime import datetime
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import is_trade_day
config = get_config()
pro = ts.pro_api(config['tushare']['token'])

fea_1 = {'ts_code': 'ts_code',
          'code': 'symbol',
          'name': 'name',
          'area': 'area',
          'industry': 'industry',
          'cnspell': 'cnspell',
          'market': 'market',
          'list_date': 'list_date',
          'act_name': 'act_name',
          'act_ent_type': 'act_ent_type'}
fea_2 = ['day', 'exchange']
exchange_map = {'BJ': 'BSE', 'SH': 'SSE', 'SZ': 'SZSE'}

def main(args):
    # tushare数据
    df = pro.stock_basic()
    print('-' * 100)
    print(df.iloc[0])
    print('股票数：{}'.format(len(df)))
    conn = mysql_connect()

    data = []
    for index, row in df.iterrows():
        tmp = {}
        tmp['day'] = args.date
        for f in fea_1.keys():
            v = row[fea_1[f]]
            if f == 'ts_code':
                suffix = v.split('.')[1]
                tmp['exchange'] = exchange_map[suffix]
            if f == 'list_date':
                v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
            if pd.isna(v):
                v = None
            tmp[f] = v
        data.append(tmp)

    print('-' * 100)
    print("data len: {}".format(len(data)))
    fea_merge = list(fea_1.keys()) + fea_2
    fea_merge_2 = ['%({})s'.format(f) for f in fea_merge]
    sql = '''
        INSERT INTO stock_info({}) VALUES ({})
    '''.format(','.join(fea_merge), ','.join(fea_merge_2))
    with conn.cursor() as cursor:
        try:
            s = cursor.mogrify(sql, data[0])
            print(s)
            cursor.executemany(sql, data)
            conn.commit()
        except Exception as e:
            print('except: {}'.format(e))
            conn.rollback()  # 回滚
            exit(1)
        finally:
            conn.close()
    print('写入完成!!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-06-30', help='取数日期')
    args = parser.parse_args()
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    print(args)
    t = time.time()
    main(args)
    print('耗时：{}s'.format(round(time.time() - t, 4)))

