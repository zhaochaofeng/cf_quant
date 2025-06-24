'''
    功能：聚宽平台数据导入mysql
    数据：市值数据表 valuation
'''
import yaml
import time
import pymysql
import argparse
import pandas as pd
from jqdatasdk import *

# 特征
features = ['code', 'day', 'capitalization', 'circulating_cap', 'market_cap',
           'circulating_market_cap', 'turnover_ratio', 'pe_ratio', 'pe_ratio_lyr',
           'pb_ratio', 'ps_ratio', 'pcf_ratio']

def parse_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        # 解析为字典
        return config
def main(args):
    # 聚宽账号
    auth(config['joinqaunt']['username'], config['joinqaunt']['password'])

    # 股票code
    stocks_info = get_all_securities(types=['stock'], date=args.date)
    stocks = stocks_info.index.tolist()
    print('股票数：{}'.format(len(stocks)))

    # 获取市值数据
    q = query(valuation).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=args.date)
    print('-' * 100)
    print(df.iloc[0])

    conn = pymysql.connect(host=config['mysql']['host'],
                         user=config['mysql']['user'],
                         password=config['mysql']['password'],
                         db=config['mysql']['db'],
                         charset='utf8')
    count = 0
    data = []
    for index, row in df.iterrows():
        # args = [','.join(features)]
        args = []
        for f in features:
            v = row[f]
            # pd中的nan需要转化为python的None
            if pd.isna(v):
                v = None
            args.append(v)
        data.append(args)
        count += 1
    print('-' * 100)
    print("data len: {}".format(len(data)))
    sql = '''
        INSERT INTO valuation({})
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''.format(','.join(features))

    with conn.cursor() as cursor:
        try:
            s = cursor.mogrify(sql, data[0])
            print(s)
            cursor.executemany(sql, data)
            conn.commit()
        except Exception as e:
            print(e)
            conn.rollback()  # 回滚
            exit(1)
        finally:
            conn.close()
        print('写入完成，数据条数：{}'.format(count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-03-20')
    parser.add_argument('--path_config', type=str, default='../../config.yaml')
    args = parser.parse_args()
    print(args)

    t = time.time()
    config = parse_config(args.path_config)
    main(args)
    print('耗时：{}s'.format(round(time.time() - t, 4)))
