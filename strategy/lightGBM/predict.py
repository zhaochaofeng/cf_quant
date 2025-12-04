'''
    加载模型，进行预测
'''
import time
from datetime import datetime
import traceback
from collections import defaultdict

import fire
import pandas as pd
import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.expm import MLflowExpManager

from utils import MySQLDB
from utils import get_n_nexttrade_day
from utils import send_email
from utils import LoggerFactory
from utils import redis_connect
from utils import CStd, CMean

class Predict:
    def __init__(
        self,
        provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
        uri: str = './mlruns',
        exp_name: str = 'lightgbm_alpha',
        instruments: str = 'csi300',
        horizon: list = None,
        start_date: str = None,
        end_date: str = None,
        is_mysql: bool = True,
        is_redis: bool = True
    ):
        self.provider_uri = provider_uri
        self.uri = uri
        self.start_date = start_date
        self.end_date = end_date
        self.exp_name = exp_name
        self.instruments = instruments
        self.is_mysql = is_mysql
        self.is_redis = is_redis
        if horizon is None:
            horizon = [1]
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        self.horizon = horizon
        self.start_date = start_date
        self.end_date = end_date
        self.model_info = {}
        self.preds = {}
        self.logger = LoggerFactory.get_logger(__name__)
        self.logger.info('\n\n\n{}\n{}: {}\n'.format('》' * 100, 'Predict', self.exp_name))
        self.initialize()

    def initialize(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'initialize ...'))
        try:
            self.logger.info('\n{}\n{}'.format('=' * 100, 'qlib init ...'))
            _setup_kwargs = {'custom_ops': [CStd, CMean]}
            qlib.init(provider_uri=self.provider_uri, **_setup_kwargs)

            # 加载实验记录
            exp_manager = MLflowExpManager(uri=self.uri, default_exp_name='default_exp')
            for h in self.horizon:
                exp_name = f'{self.exp_name}_h{h}'
                self.logger.info('exp_name: {}'.format(exp_name))
                self.model_info[exp_name] = {}
                exp = exp_manager.get_exp(experiment_name=exp_name)
                if not exp:
                    raise ValueError(f"找不到Name为 {exp_name} 的实验记录")

                # 获取最新的在线模型记录器
                recorders = exp.list_recorders()
                online_recorders = []
                for rid, rec in recorders.items():
                    if rec.status != 'FINISHED':
                        continue
                    # 检查上线状态
                    # tags = rec.list_tags()
                    # if tags.get('online_status') == 'online':
                    #     online_recorders.append(rec)
                    online_recorders.append(rec)

                if not online_recorders:
                    raise ValueError(f"实验 {exp_name} 中没有找到在线模型记录")

                # 选择最新的在线模型
                newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)
                # 加载模型
                model = newest_recorder.load_object("params.pkl")
                # 加载dataset
                dataset = newest_recorder.load_object("dataset")
                self.model_info[exp_name]['model'] = model
                self.model_info[exp_name]['dataset'] = dataset

        except Exception as e:
            raise Exception(f"模型初始化失败: {str(e)}")

        assert len(self.horizon) == len(self.model_info)
        self.logger.info(self.model_info)
        self.logger.info('初始化成功 !')

    def predict(self, start_date, end_date):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'predict ...'))
        for exp_name in self.model_info:
            self.logger.info('exp_name: {}'.format(exp_name))
            model = self.model_info[exp_name]['model']
            dataset = self.model_info[exp_name]['dataset']
            segments = {"test": (start_date, end_date)}
            dataset.config(
                handler_kwargs={"start_time": start_date, "end_time": end_date},
                segments=segments)
            # 准备数据
            dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
            self.logger.info('\n{}\n{}'.format('-' * 50, dataset.prepare(segments='test')))

            # 预测
            df = model.predict(dataset)
            if df is None or df.empty:
                raise ValueError(f"predict df is empty: {exp_name} ")
            df = df.reset_index()
            df.columns = ['day', 'qlib_code', 'score']
            self.preds[exp_name] = df
        self.logger.info(self.preds)

    def merge_preds(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'merge_preds ...'))
        merged_list = []
        for exp_name in self.preds:
            hr = int(exp_name.replace(f'{self.exp_name}_h', ''))
            df = self.preds[exp_name]
            # 获取唯一日期，避免重复请求
            unique_day = df['day'].unique()
            day_mapping = {}
            # 对每个唯一日期仅请求一次
            for day in unique_day:
                time.sleep(60/500)  # 1min 最多请求500次
                date_str = pd.to_datetime(day).strftime('%Y-%m-%d')
                new_date = get_n_nexttrade_day(date_str, hr)
                day_mapping[day] = new_date
            # 使用映射表更新所有日期
            df['day'] = df['day'].map(day_mapping)
            df['day'] = pd.to_datetime(df['day'])

            merged_list.append(df)
        merged = pd.concat(merged_list, axis=0, ignore_index=True)
        return merged

    def write_to_mysql(self, merged: pd.DataFrame):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'write_to_mysql ...'))
        data = []
        columns = merged.columns.to_list()
        for index, row in merged.iterrows():
            tmp = {}
            for c in columns:
                tmp[c] = row[c]
            tmp['model'] = self.exp_name
            tmp['instruments'] = self.instruments
            data.append(tmp)
        self.logger.info('data len: {}'.format(len(data)))

        columns = columns + ['model', 'instruments']
        columns_format = ["%({})s".format(c) for c in columns]
        sql = '''
            INSERT INTO monitor_return_rate({}) VALUES ({})
            ON DUPLICATE KEY UPDATE score=VALUES(score)
        '''.format(','.join(columns), ','.join(columns_format))
        with MySQLDB() as db:
            db.executemany(sql, data)

        self.logger.info('write complete !!!')

    def write_to_redis(self, df: pd.DataFrame) -> None:
        self.logger.info('\n{}\n{}'.format('=' * 100, 'write_to_redis ...'))
        r = redis_connect()
        data = defaultdict(dict)
        for idx, row in df.iterrows():
            day = row['day'].strftime('%Y-%m-%d')
            data[day][row['qlib_code']] = row['score']
        for key, value in data.items():
            key = '{}:{}:{}'.format(self.exp_name, self.instruments, key)
            self.logger.info('key: {}'.format(key))
            r.hset(key, mapping=value)
        self.logger.info('write complete !!!')

    def main(self):
        try:
            t = time.time()
            # 预测
            self.predict(self.start_date, self.end_date)
            # 合并预测结果
            merged = self.merge_preds()
            # 写入mysql
            if self.is_mysql:
                self.write_to_mysql(merged)
            # 写入redis
            if self.is_redis:
                self.write_to_redis(merged)
            self.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
        except Exception as e:
            erro_info = traceback.format_exc()
            self.logger.error(erro_info)
            send_email(f'Strategy:predict:{self.exp_name}-{self.instruments}', erro_info)


if __name__ == '__main__':
    fire.Fire(Predict)
    '''
        python predict.py main --start_date 2025-09-19 --end_date 2025-09-19 --horizon "[1,2]"
    '''
