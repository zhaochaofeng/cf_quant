'''
    数据：tushare平台的技术指标(因子)
    描述：stk_factor_pro接口每分钟最多请求30次，每次1w条数据
'''

import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import tushare_pro, is_trade_day
from utils.utils import get_n_pretrade_day

config = get_config()
pro = tushare_pro()
batch_size = 10000

fea_to_from = {
    'ts_code': 'ts_code',
    'ann_date': 'ann_date',
    'end_date': 'end_date',
    'eps': 'eps',
    'dt_eps': 'dt_eps',
    'total_revenue_ps': 'total_revenue_ps',
    'revenue_ps': 'revenue_ps',
    'capital_rese_ps': 'capital_rese_ps',
    'surplus_rese_ps': 'surplus_rese_ps',
    'undist_profit_ps': 'undist_profit_ps',
    'extra_item': 'extra_item',
    'profit_dedt': 'profit_dedt',
    'gross_margin': 'gross_margin',
    'current_ratio': 'current_ratio',
    'quick_ratio': 'quick_ratio',
    'cash_ratio': 'cash_ratio',
    'ar_turn': 'ar_turn',
    'ca_turn': 'ca_turn',
    'fa_turn': 'fa_turn',
    'assets_turn': 'assets_turn',
    'op_income': 'op_income',
    'ebit': 'ebit',
    'ebitda': 'ebitda',
    'fcff': 'fcff',
    'fcfe': 'fcfe',
    'current_exint': 'current_exint',
    'noncurrent_exint': 'noncurrent_exint',
    'interestdebt': 'interestdebt',
    'netdebt': 'netdebt',
    'tangible_asset': 'tangible_asset',
    'working_capital': 'working_capital',
    'networking_capital': 'networking_capital',
    'invest_capital': 'invest_capital',
    'retained_earnings': 'retained_earnings',
    'diluted2_eps': 'diluted2_eps',
    'bps': 'bps',
    'ocfps': 'ocfps',
    'retainedps': 'retainedps',
    'cfps': 'cfps',
    'ebit_ps': 'ebit_ps',
    'fcff_ps': 'fcff_ps',
    'fcfe_ps': 'fcfe_ps',
    'netprofit_margin': 'netprofit_margin',
    'grossprofit_margin': 'grossprofit_margin',
    'cogs_of_sales': 'cogs_of_sales',
    'expense_of_sales': 'expense_of_sales',
    'profit_to_gr': 'profit_to_gr',
    'saleexp_to_gr': 'saleexp_to_gr',
    'adminexp_of_gr': 'adminexp_of_gr',
    'finaexp_of_gr': 'finaexp_of_gr',
    'impai_ttm': 'impai_ttm',
    'gc_of_gr': 'gc_of_gr',
    'op_of_gr': 'op_of_gr',
    'ebit_of_gr': 'ebit_of_gr',
    'roe': 'roe',
    'roe_waa': 'roe_waa',
    'roe_dt': 'roe_dt',
    'roa': 'roa',
    'npta': 'npta',
    'roic': 'roic',
    'roe_yearly': 'roe_yearly',
    'roa2_yearly': 'roa2_yearly',
    'debt_to_assets': 'debt_to_assets',
    'assets_to_eqt': 'assets_to_eqt',
    'dp_assets_to_eqt': 'dp_assets_to_eqt',
    'ca_to_assets': 'ca_to_assets',
    'nca_to_assets': 'nca_to_assets',
    'tbassets_to_totalassets': 'tbassets_to_totalassets',
    'int_to_talcap': 'int_to_talcap',
    'eqt_to_talcapital': 'eqt_to_talcapital',
    'currentdebt_to_debt': 'currentdebt_to_debt',
    'longdeb_to_debt': 'longdeb_to_debt',
    'ocf_to_shortdebt': 'ocf_to_shortdebt',
    'debt_to_eqt': 'debt_to_eqt',
    'eqt_to_debt': 'eqt_to_debt',
    'eqt_to_interestdebt': 'eqt_to_interestdebt',
    'tangibleasset_to_debt': 'tangibleasset_to_debt',
    'tangasset_to_intdebt': 'tangasset_to_intdebt',
    'tangibleasset_to_netdebt': 'tangibleasset_to_netdebt',
    'ocf_to_debt': 'ocf_to_debt',
    'turn_days': 'turn_days',
    'roa_yearly': 'roa_yearly',
    'roa_dp': 'roa_dp',
    'fixed_assets': 'fixed_assets',
    'profit_to_op': 'profit_to_op',
    'q_saleexp_to_gr': 'q_saleexp_to_gr',
    'q_gc_to_gr': 'q_gc_to_gr',
    'q_roe': 'q_roe',
    'q_dt_roe': 'q_dt_roe',
    'q_npta': 'q_npta',
    'q_ocf_to_sales': 'q_ocf_to_sales',
    'basic_eps_yoy': 'basic_eps_yoy',
    'dt_eps_yoy': 'dt_eps_yoy',
    'cfps_yoy': 'cfps_yoy',
    'op_yoy': 'op_yoy',
    'ebt_yoy': 'ebt_yoy',
    'netprofit_yoy': 'netprofit_yoy',
    'dt_netprofit_yoy': 'dt_netprofit_yoy',
    'ocf_yoy': 'ocf_yoy',
    'roe_yoy': 'roe_yoy',
    'bps_yoy': 'bps_yoy',
    'assets_yoy': 'assets_yoy',
    'eqt_yoy': 'eqt_yoy',
    'tr_yoy': 'tr_yoy',
    'or_yoy': 'or_yoy',
    'q_sales_yoy': 'q_sales_yoy',
    'q_op_qoq': 'q_op_qoq',
    'equity_yoy': 'equity_yoy'
}

def parse_line(row, fea_to_from):
    ''' 解析数据 '''
    tmp = {'day': args.date}
    for f in fea_to_from.keys():
        try:
            v = row[fea_to_from[f]]
            if f in ['ann_date', 'end_date']:
                # 日期格式转换
                v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
            if pd.isna(v):
                v = None
            tmp[f] = v
        except Exception as e:
            print('except: {}'.format(e))
    return tmp

def request_from_tushare(date):
    # 从tushare API获取数据
    print('-' * 100)
    print('从tushare API获取数据...')
    end_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=300)).strftime('%Y%m%d')
    print('start_date: {}, end_date: {}'.format(start_date, end_date))

    info = pro.fina_indicator_vip(start_date=start_date, end_date=end_date)
    # 由于tushare表中存在数据更新操作，一只股票可能有多条数据，选择缺失值最少的那一条
    info = info.groupby(['ts_code', 'end_date']).apply(lambda x: x.loc[x.isna().sum(axis=1).idxmin()]).reset_index(drop=True)
    print(info.head(100))
    data = []
    for index, row in info.iterrows():
        try:
            tmp = parse_line(row, fea_to_from)
            data.append(tmp)
        except Exception as e:
            print('except: {}'.format(e))
            continue
    return data

def write_to_mysql(data):
    print('-' * 100)
    print('导入msyql ...')
    conn = mysql_connect()

    with conn.cursor() as cursor:
        cursor.execute('delete from fina_indicator;')
        conn.commit()
        print('清空历史数据！！！')

    with conn.cursor() as cursor:
        feas = list(fea_to_from.keys())
        feas.insert(1, 'day')
        fea_format = ['%({})s'.format(f) for f in feas]

        sql = '''
                    INSERT INTO fina_indicator({}) VALUES ({})
                '''.format(','.join(feas), ','.join(fea_format))

        batch_num = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
        print('batch_num: {}'.format(batch_num))
        for i in range(int(batch_num)):
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

    # 1、获取交易数据
    data = request_from_tushare(args.date)
    print('数据请求耗时：{}s'.format(round(time.time()-t, 4)))
    t = time.time()
    print('data len: {}'.format(len(data)))
    if len(data) < 100:
        print('tushare 数据请求失败 ！！！')
        exit(1)

    # 2、写入mysql
    write_to_mysql(data)
    print('数据写入耗时：{}s'.format(round(time.time() - t, 4)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-07-11', help='取数日期')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('总耗时：{}s'.format(round(time.time() - t, 4)))
