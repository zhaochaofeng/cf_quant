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
    'day': 'trade_date',
    'lowdays': 'lowdays',
    'topdays': 'topdays',
    'downdays': 'downdays',
    'updays': 'updays',
    'asi_qfq': 'asi_qfq',
    'asit_qfq': 'asit_qfq',
    'atr_qfq': 'atr_qfq',
    'bbi_qfq': 'bbi_qfq',
    'bias1_qfq': 'bias1_qfq',
    'bias2_qfq': 'bias2_qfq',
    'bias3_qfq': 'bias3_qfq',
    'boll_lower_qfq': 'boll_lower_qfq',
    'boll_mid_qfq': 'boll_mid_qfq',
    'boll_upper_qfq': 'boll_upper_qfq',
    'brar_ar_qfq': 'brar_ar_qfq',
    'brar_br_qfq': 'brar_br_qfq',
    'cci_qfq': 'cci_qfq',
    'cr_qfq': 'cr_qfq',
    'dfma_dif_qfq': 'dfma_dif_qfq',
    'dfma_difma_qfq': 'dfma_difma_qfq',
    'dmi_adx_qfq': 'dmi_adx_qfq',
    'dmi_adxr_qfq': 'dmi_adxr_qfq',
    'dmi_mdi_qfq': 'dmi_mdi_qfq',
    'dmi_pdi_qfq': 'dmi_pdi_qfq',
    'dpo_qfq': 'dpo_qfq',
    'madpo_qfq': 'madpo_qfq',
    'ema_qfq_10': 'ema_qfq_10',
    'ema_qfq_20': 'ema_qfq_20',
    'ema_qfq_250': 'ema_qfq_250',
    'ema_qfq_30': 'ema_qfq_30',
    'ema_qfq_5': 'ema_qfq_5',
    'ema_qfq_60': 'ema_qfq_60',
    'ema_qfq_90': 'ema_qfq_90',
    'emv_qfq': 'emv_qfq',
    'maemv_qfq': 'maemv_qfq',
    'expma_12_qfq': 'expma_12_qfq',
    'expma_50_qfq': 'expma_50_qfq',
    'kdj_qfq': 'kdj_qfq',
    'kdj_d_qfq': 'kdj_d_qfq',
    'kdj_k_qfq': 'kdj_k_qfq',
    'ktn_down_qfq': 'ktn_down_qfq',
    'ktn_mid_qfq': 'ktn_mid_qfq',
    'ktn_upper_qfq': 'ktn_upper_qfq',
    'ma_qfq_10': 'ma_qfq_10',
    'ma_qfq_20': 'ma_qfq_20',
    'ma_qfq_250': 'ma_qfq_250',
    'ma_qfq_30': 'ma_qfq_30',
    'ma_qfq_5': 'ma_qfq_5',
    'ma_qfq_60': 'ma_qfq_60',
    'ma_qfq_90': 'ma_qfq_90',
    'macd_qfq': 'macd_qfq',
    'macd_dea_qfq': 'macd_dea_qfq',
    'macd_dif_qfq': 'macd_dif_qfq',
    'mass_qfq': 'mass_qfq',
    'ma_mass_qfq': 'ma_mass_qfq',
    'mfi_qfq': 'mfi_qfq',
    'mtm_qfq': 'mtm_qfq',
    'mtmma_qfq': 'mtmma_qfq',
    'obv_qfq': 'obv_qfq',
    'psy_qfq': 'psy_qfq',
    'psyma_qfq': 'psyma_qfq',
    'roc_qfq': 'roc_qfq',
    'maroc_qfq': 'maroc_qfq',
    'rsi_qfq_12': 'rsi_qfq_12',
    'rsi_qfq_24': 'rsi_qfq_24',
    'rsi_qfq_6': 'rsi_qfq_6',
    'taq_down_qfq': 'taq_down_qfq',
    'taq_mid_qfq': 'taq_mid_qfq',
    'taq_up_qfq': 'taq_up_qfq',
    'trix_qfq': 'trix_qfq',
    'trma_qfq': 'trma_qfq',
    'vr_qfq': 'vr_qfq',
    'wr_qfq': 'wr_qfq',
    'wr1_qfq': 'wr1_qfq',
    'xsii_td1_qfq': 'xsii_td1_qfq',
    'xsii_td2_qfq': 'xsii_td2_qfq',
    'xsii_td3_qfq': 'xsii_td3_qfq',
    'xsii_td4_qfq': 'xsii_td4_qfq'
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

def request_from_tushare(date, count=1):
    # 从tushare API获取数据
    print('-' * 100)
    print('从tushare API获取数据...')
    end_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y%m%d')
    start_date = datetime.strptime(get_n_pretrade_day(date, count - 1), '%Y-%m-%d').strftime('%Y%m%d')
    print('start_date: {}, end_date: {}'.format(start_date, end_date))

    # 交易日期列表
    date_list = pro.trade_cal(start_date=start_date, end_date=end_date, is_open='1')['cal_date'].values.tolist()
    print('date_list: {}'.format(date_list))
    print('date_list len: {}'.format(len(date_list)))

    data = []
    # 按日期请求
    for i, date in enumerate(date_list):
        # time.sleep(0.1)  # API调用频次：1min不超过30次
        if (i + 1) % 10 == 0:
            print('requested num: {}'.format(i + 1))
        info = pro.stk_factor_pro(trade_date=date)
        if info is None:
            continue
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
        fea_format = ['%({})s'.format(f) for f in list(fea_to_from.keys())]
        fea_keys = ','.join(list(fea_to_from.keys()))

        sql = '''
                    INSERT INTO tsfactor({}) VALUES ({})
                '''.format(fea_keys, ','.join(fea_format))

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
    data = request_from_tushare(args.date, args.count)
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
    parser.add_argument('--date', type=str, default='2025-07-10', help='取数日期')
    parser.add_argument('--count', type=int, default=1, help='取数天数，从date往前取')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('总耗时：{}s'.format(round(time.time() - t, 4)))
