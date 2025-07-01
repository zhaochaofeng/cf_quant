import time
import argparse
import pandas as pd
import tushare as ts
from datetime import datetime
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import tushare_pro, is_trade_day
config = get_config()
pro = tushare_pro()

fea_to_from = {
     'ts_code': 'ts_code',
     'day': 'trade_date',
     'close': 'close',
     'turnover_rate': 'turnover_rate',
     'turnover_rate_f': 'turnover_rate_f',
     'volume_ratio': 'volume_ratio',
     'pe': 'pe',
     'pe_ttm': 'pe_ttm',
     'pb': 'pb',
     'ps': 'ps',
     'ps_ttm': 'ps_ttm',
     'dv_ratio': 'dv_ratio',
     'dv_ttm': 'dv_ttm',
     'total_share': 'total_share',
     'float_share': 'float_share',
     'free_share': 'free_share',
     'total_mv': 'total_mv',
     'circ_mv': 'circ_mv'
}

def main(args):
    # tushare数据
    ts_codes = pro.stock_basic()['ts_code'].values.tolist()
    print('-' * 100)
    print('股票数：{}'.format(len(ts_codes)))
    conn = mysql_connect()

    date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y%m%d')
    print('date: {}'.format(date))
    data = []
    for i, code in enumerate(ts_codes):
        time.sleep(0.1)
        if (i+1) % 100 == 0:
            print('process: {}'.format(i+1))
        tmp = {}
        info = pro.daily_basic(ts_code=code, start_date=date, end_date=date)
        if info.empty:
            continue
        for f in fea_to_from.keys():
            try:
                v = info[fea_to_from[f].lower()][0]
                if pd.isna(v):
                    v = None
                tmp[f] = v
            except Exception as e:
                print('except: {}'.format(e))
                print('code: {}'.format(code))
                print(info)
                continue
        data.append(tmp)

    print('-' * 100)
    print("data len: {}".format(len(data)))
    fea_format = ['%({})s'.format(f) for f in list(fea_to_from.keys())]
    sql = '''
        INSERT INTO valuation_tushare({}) VALUES ({})
    '''.format(','.join(list(fea_to_from.keys())), ','.join(fea_format))
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
    parser.add_argument('--date', type=str, default='2025-07-01', help='取数日期')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('耗时：{}s'.format(round(time.time() - t, 4)))
