'''
    功能：由于复权因子在除权除息日附近可能变动，每天检查历史复权因子，如果发生变动，则更新历史数据
'''
import fire
import time
import pandas as pd
import traceback
from datetime import datetime
from utils.utils import sql_engine
from utils.utils import tushare_pro
from utils.utils import get_trade_cal_inter
from utils.utils import send_email
from utils.utils import mysql_connect
from utils.utils import is_trade_day

engine = sql_engine()
pro = tushare_pro()

def get_factor_ts(start_date, end_date):
    print('-' * 100)
    print('get_factor_ts ...')
    sql = ''' select ts_code, day, adj_factor from trade_daily_ts where day>='{}' and day<='{}' '''.format(start_date, end_date)
    print(sql)
    factor_old = pd.read_sql(sql, engine)
    factor_new = pd.DataFrame()
    for date in get_trade_cal_inter(start_date, end_date):
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        tmp = pro.adj_factor(start_date=date, end_date=date)
        factor_new = pd.concat([factor_new, tmp], axis=0, join='outer')
    factor_new.rename({'trade_date': 'day'}, axis=1, inplace=True)
    factor_new['day'] = pd.to_datetime(factor_new['day'], format='%Y%m%d')

    return factor_old, factor_new

def repair_factor(factor_old, factor_new):
    print('-' * 100)
    print('repair_factor ...')
    conn = mysql_connect()
    diff = factor_old.ne(factor_new)
    mask = diff.any(axis=1)
    index_ne = diff.index[mask]
    if index_ne.empty:
        return []

    res = []
    for index in index_ne:
        try:
            old_f = factor_old.loc[index]['adj_factor']
        except:
            old_f = 'NaN'
        try:
            new_f = factor_new.loc[index]['adj_factor']
        except:
            new_f = 'NaN'

        res.append('{}: [old_factor: {};  new_factor: {}]'.format(index, old_f, new_f))
        if old_f != 'NaN' and new_f != 'NaN' and old_f != new_f:
            try:
                with conn.cursor() as cursor:
                    sql = ''' update trade_daily_ts set adj_factor=%s where ts_code=%s and day=%s '''
                    print(cursor.mogrify(sql, (new_f, index[0], index[1].strftime('%Y-%m-%d'))))
                    cursor.execute(sql, (new_f, index[0], index[1].strftime('%Y-%m-%d')))
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise Exception('mysql update factor fail: '.format(e))
    return res

def main(start_date, end_date, source='ts'):
    '''
    :param source: data source. ['ts', 'bao']
    '''
    if not is_trade_day(end_date):
        print('非交易日，退出！！！')
        exit(0)

    try:
        if source == 'ts':
            factor_old, factor_new = get_factor_ts(start_date, end_date)
        elif source == 'bao':
            factor_old, factor_new = None, None
        else:
            raise ValueError('param source must be ts or bao !')

        if factor_old.empty or factor_old.empty:
            raise Exception('factor_old or factor_new is empty !')

        factor_old.set_index(keys=['ts_code', 'day'], inplace=True)
        factor_new.set_index(keys=['ts_code', 'day'], inplace=True)
        print('factor_old shape: {}'.format(factor_old.shape))
        print('factor_new shape: {}'.format(factor_new.shape))
        print(factor_old.head())
        print(factor_new.head())
        factor_new = factor_new.reindex(factor_old.index)
        print('factor_new shape: {}'.format(factor_new.shape))
        res = repair_factor(factor_old, factor_new)
        if len(res) > 0:
            send_email(f'Data:trade_daily:update_factor:{source}', '\n'.join(res))
        else:
            print('Check no problem !!!')
    except:
        error_info = traceback.format_exc()
        send_email(f'Data:trade_daily:update_factor:{source}', error_info)

if __name__ == '__main__':
    t = time.time()
    fire.Fire(main)
    print('耗时：{}s'.format(round(time.time()-t, 4)))