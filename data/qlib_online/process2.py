'''
    功能：将mysql中数据处理成qlib格式数据
'''

import os
import time
import fire
import shutil
import pandas as pd
import traceback
from utils import (
    LoggerFactory,
    tushare_pro, sql_engine,
    is_trade_day, send_email
)
import warnings
warnings.filterwarnings("ignore")

class Processor:
    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq_t',
                 path_in: str = None,            # 离线数据路径
                 start_date: str = None,         # 数据起始日期
                 end_date: str = None,           # 数据终止日期
                 is_offline: bool = True,       # 是否处理批量导出的离线数据
                 index_list: list = None,        # 指数code
                 fq: str = 'hfq',                # 复权方式。hfq: 后复权; qfq: 前复权; None: 不复权
                 columns: list = None
                 ):
        if columns is None:
            columns = ['ts_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'factor', 'change']
        if index_list is None:
            index_list = ['000300.SH', '000905.SH', '000903.SH']
        self.provider_uri = provider_uri
        self.path_in = path_in
        self.start_date = start_date
        self.end_date = end_date
        self.is_offline = is_offline
        self.index_list = index_list
        self.fq = fq
        self.columns = columns

        if self.start_date is None or self.end_date is None or self.start_date > self.end_date:
            raise ValueError('start_date and end_date cannot be None, start_date must be less then or equal to end_date')
        if self.is_offline and self.path_in is None:
            raise ValueError("When is_offline is True, path_in cannot be None")
        self.logger = LoggerFactory.get_logger(__name__)

    def load_stock(self) -> pd.DataFrame:
        ''' 加载股票数据 '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'load_stock ...'))
        if self.is_offline:
            df = pd.read_csv(self.path_in, sep='\t')
        else:
            engine = sql_engine()
            sql = '''
            select ts_code, day as date, open, close, high, low, vol, amount, adj_factor 
            from trade_daily_ts where day>='{}' and day<='{}';
            '''.format(self.start_date, self.end_date)
            self.logger.info('{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
            df = pd.read_sql(sql, engine)
            # 将date字段从datetime.date转化为str格式
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
            df.to_csv(os.path.join(self.provider_uri, 'custom_{}_{}.csv'.format(self.start_date, self.end_date)), sep='\t', index=False)

        if df.empty:
            err_msg = 'df is empty !'
            self.logger.error(err_msg)
            raise Exception(err_msg)
        rename_dic = {'vol': 'volume', 'adj_factor': 'factor'}
        df.rename(columns=rename_dic, inplace=True)
        self.logger.info('df shape: {}'.format(df.shape))
        return df

    def load_index(self, codes: list) -> pd.DataFrame:
        ''' 加载指数数据 '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'load_index ...'))
        pro = tushare_pro()

        start_date = self.start_date.replace('-', '')
        end_date = self.end_date.replace('-', '')

        df_list = []
        rename_dic = {'trade_date': 'date', 'vol': 'volume'}
        for code in codes:
            try:
                self.logger.info('index code: {}'.format(code))
                tmp_df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
                tmp_df.rename(columns=rename_dic, inplace=True)
                tmp_df['date'] = pd.to_datetime(tmp_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                tmp_df['factor'] = 1
                df_list.append(tmp_df)
            except Exception as e:
                err_msg = f'request from tushare failed：{e}'
                self.logger.error(err_msg)
                raise ValueError(err_msg)
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df = df[['ts_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'factor']]
        if df.empty:
            err_msg = 'df is empty !'
            self.logger.error(err_msg)
            raise Exception(err_msg)
        self.logger.info('df shape: {}'.format(df.shape))
        return df

    def trans_fq(self, df):
        ''' 复权计算 '''
        self.logger.info('\n{}\n{}:{} ...'.format('=' * 100, 'trans_fq', self.fq))
        def hfq(group):
            ''' 后复权 '''
            # 对open, close, high, low, volume 复权
            group['open'] = group['open'] * group['factor']
            group['close'] = group['close'] * group['factor']
            group['high'] = group['high'] * group['factor']
            group['low'] = group['low'] * group['factor']
            group['volume'] = group['volume'] / group['factor']
            return group

        def qfq(group):
            ''' 前复权 '''
            filtered_group = group[group['date'] == self.end_date]
            if filtered_group.empty:
                return None
            now_factor = filtered_group['factor'].iloc[0]
            # 对open, close, high, low, volume复权
            group['open'] = group['open'] * group['factor'] / now_factor
            group['close'] = group['close'] * group['factor'] / now_factor
            group['high'] = group['high'] * group['factor'] / now_factor
            group['low'] = group['low'] * group['factor'] / now_factor
            group['volume'] = group['volume'] * now_factor / group['factor']
            return group

        if self.fq == 'hfq':
            df = df.groupby(by='ts_code', group_keys=False).apply(hfq)
        elif self.fq == 'qfq':
            df = df.groupby(by='ts_code', group_keys=False).apply(qfq)
        self.logger.info('df shape: {}'.format(df.shape))
        return df

    def split_stock_data(self, df):
        """ 按股票代码分割成单独的CSV文件 """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'split_stock_data...'))
        output_dir = os.path.join(self.provider_uri, 'out_{}_{}'.format(self.start_date, self.end_date))
        output_dir = os.path.expanduser(output_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 按股票代码分组并保存
        for i, (ts_code, group) in enumerate(df.groupby('ts_code')):
            code = '{}{}'.format(ts_code.split('.')[1], ts_code.split('.')[0])
            output_file = os.path.join(output_dir, f"{code}.csv")
            group.to_csv(output_file, index=False, columns=self.columns[1:])
            if (i+1) % 100 == 0:
                self.logger.info(f"processed : {i+1} ")
                # self.logger.info(f"生成: {output_file} ({len(group)}行)")

    def main(self):
        try:
            t = time.time()
            if not is_trade_day(self.end_date):
                self.logger.warning('is not trade day: {}'.format(self.end_date))
                exit(0)

            df = self.load_stock()
            index_df = self.load_index(self.index_list)
            merged = pd.concat([df, index_df], axis=0, join='outer', ignore_index=True)
            self.logger.info('merged shape: {}'.format(merged.shape))
            merged_fq = self.trans_fq(merged)

            # 单位转化
            merged_fq['volume'] = merged_fq['volume'] * 100   # 单位股
            merged_fq['amount'] = merged_fq['amount'] * 1000  # 单位元

            # change字段
            def get_change(group):
                group.sort_values(by='date', inplace=True)
                group['change'] = (group['close'] - group['close'].shift(1)) / group['close'].shift(1)
                return group
            merged_fq = merged_fq.groupby(by='ts_code', group_keys=False).apply(get_change)
            merged_fq = merged_fq.round({'open': 2, 'close': 2, 'high': 2, 'low': 2, 'volume': 2, 'amount': 3, 'change': 4, 'factor': 4})
            # 保存中间数据
            # merged_fq.to_csv(os.path.join(self.provider_uri, 'hfq_{}_{}.csv'.format(self.start_date, self.end_date)), index=False)
            self.logger.info(merged_fq.head())
            self.split_stock_data(merged_fq)
            self.logger.info('耗时：{} s'.format(round(time.time() - t, 4)))
        except:
            error_msg = traceback.format_exc()
            self.logger.error(error_msg)
            send_email('Data:qlib_online:process', error_msg)

if __name__ == '__main__':
    fire.Fire(Processor)
    '''
        批量离线: 
        python process.py main \
            --start_date 2025-01-01 \
            --end_date 2025-08-01 \
            --is_offline True \
            --path_in ~/.qlib/qlib_data/custom_data_hfq_tmp/custom_2025-01-01_2025-08-01.csv

        每日更新(暂不可用)：
        python process.py main --start_date 2025-08-01 --end_date 2025-08-01
    '''
