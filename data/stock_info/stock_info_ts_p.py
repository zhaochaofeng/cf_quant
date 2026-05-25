'''
    股票基本上信息表：stock_info_ts
'''

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import time
import traceback
from datetime import datetime
from prefect import flow
import pandas as pd
from data.process_data import Base, ts_api
from utils import tushare_pro, send_email, is_trade_day
from utils import email_send_message_flow

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'day': '',
    'name': 'name',
    'status': 'status',
    'area': 'area',
    'industry': 'industry',
    'fullname': 'fullname',
    'enname': 'enname',
    'cnspell': 'cnspell',
    'market': 'market',
    'exchange': 'exchange',
    'curr_type': 'curr_type',
    'list_date': 'list_date',
    'delist_date': 'delist_date',
    'is_hs': 'is_hs',
    'act_name': 'act_name',
    'act_ent_type': 'act_ent_type',
    'l1_code': 'l1_code',
    'l1_name': 'l1_name',
    'l2_code': 'l2_code',
    'l2_name': 'l2_name',
    'l3_code': 'l3_code',
    'l3_name': 'l3_name',
    'in_date': 'in_date',
    'out_date': 'out_date',
    'is_new': 'is_new'
}
exclude_codes = {'T00018.SH', 'TS0018.SH'}  # 噪声数据


class TSStockInfoProcessor(Base):
    ''' 股票信息数据 '''

    def __init__(self,
                 feas: dict = feas,
                 table_name: str = 'stock_info_ts',
                 now_date: str = None,
                 **kwargs
                 ):
        super().__init__(feas=feas, table_name=table_name, **kwargs)
        self.now_date = now_date if now_date else datetime.now().strftime('%Y-%m-%d')

    def fetch_data_from_api(self):
        ''' 股票基本信息 + 行业数据 '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api ...'))
        try:
            pro = tushare_pro()
            # 基本信息
            df_info = pd.DataFrame()
            status = {'L': 1, 'D': 0, 'P': 2, 'G': 3}
            for k, v in status.items():
                tmp = ts_api(pro, 'stock_basic', list_status=k,
                             fields='ts_code,name,area,industry,fullname,enname,cnspell,market,'
                                    'exchange,curr_type,list_date,delist_date,is_hs,act_name,act_ent_type')
                tmp['status'] = v
                df_info = pd.concat([df_info, tmp], ignore_index=True)
            df_info.set_index('ts_code', inplace=True)
            # 排除噪声数据
            df_info = df_info[~df_info.index.isin(exclude_codes)]
            self.logger.info('df_info shape: {}'.format(df_info.shape))
            if df_info.empty:
                err_msg = 'df_info is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)

            # 行业数据
            df_ind = pd.DataFrame()
            # 一级行业代码集合
            ind_code = pro.index_classify(level='L1', src='SW2021')['index_code'].values.tolist()
            for code in ind_code:
                tmp = ts_api(pro, 'index_member_all', l1_code=code)
                df_ind = pd.concat([df_ind, tmp], ignore_index=True)
                df_ind.drop(columns='name', inplace=True)
            # 行业信息可能变动，导致同一个ts_code对应多个行业
            df_ind.sort_values(by=['ts_code', 'in_date'], inplace=True)
            df_ind.drop_duplicates(subset=['ts_code'], keep='last', inplace=True)
            df_ind.set_index('ts_code', inplace=True)

            self.logger.info('df_ind shape: {}'.format(df_ind.shape))
            if df_ind.empty:
                err_msg = 'df_ind is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)

            print(df_info.head())
            print(df_ind.head())
            df_ind = df_ind.reindex(df_info.index)
            df_merge = pd.concat([df_info, df_ind], axis=1, join='outer')
            df_merge.reset_index(inplace=True)
            self.logger.info('df_merge shape: {}'.format(df_merge.shape))

            return df_merge
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def parse_line(self, row) -> dict:
        ''' 解析单条数据 '''
        try:
            tmp = {}
            for f in self.feas.keys():
                if f == 'day' and self.feas[f] == '':
                    v = self.now_date
                else:
                    v = row[self.feas[f]]
                    if pd.isna(v):
                        v = None
                    elif f == 'qlib_code':
                        code, suffix = v.split('.')
                        v = '{}{}'.format(suffix.upper(), code)
                    elif f in ['day', 'in_date', 'delist_date', 'list_date']:
                        # 日期格式转换
                        v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                tmp[f] = v
            return tmp
        except Exception as e:
            error_msg = 'parse_line error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def main(self) -> None:
        t = time.time()
        try:
            df = self.fetch_data_from_api()
            data = self.process(df)
            self.write_to_mysql(data)
            print('耗时：{}s'.format(round(time.time() - t, 4)))
        except:
            error_msg = traceback.format_exc()
            self.logger.error(error_msg)
            send_email('Data: stock_info_ts', error_msg)
            raise Exception(error_msg)


@flow(log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def stock_info_ts_flow(now_date: str = ''):
    '''Prefect flow: 每日定时拉取股票信息'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return
    processor = TSStockInfoProcessor(now_date=now_date)
    try:
        processor.main()
    except Exception as e:
        err_msg = f'stock_info_ts_flow({now_date}) 执行失败:\n{traceback.format_exc()}'
        print(err_msg)
        email_send_message_flow(subject='Data: stock_info_ts', msg=err_msg)
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, default='',
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        # Prefect 部署注册
        from prefect.schedules import Schedule
        from pathlib import Path

        schedule = Schedule(
            cron="1 18 * * *",
            timezone="Asia/Shanghai",
        )
        stock_info_ts_flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="stock_info_ts_prefect.py:stock_info_ts_flow",
        ).deploy(
            name="stock-info-ts-daily",
            work_pool_name="cf_quant",
            schedule=schedule,
        )
    else:
        stock_info_ts_flow(now_date=args.now_date)
