'''
    加载模型，进行预测
'''
import time
import fire
import pandas as pd
import qlib
from typing import Union
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.expm import MLflowExpManager
from utils.utils import get_n_nexttrade_day
from utils.utils import mysql_connect
from utils.utils import send_email
import traceback

class Predict:
    def __init__(
        self,
        provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
        uri: str = './mlruns',
        start_date: str = None,
        end_date: str = None,
        horizon=None
    ):
        if horizon is None:
            horizon = [1]
        elif isinstance(horizon, str):
            horizon = [int(n) for n in horizon.split(',')]
        elif isinstance(horizon, Union[tuple, list]):
            horizon = [int(n) for n in horizon]
        elif isinstance(horizon, int):
            horizon = [horizon]
        else:
            raise ValueError('horizon type error。{}:{}'.format(horizon, type(horizon)))

        self.provider_uri = provider_uri
        self.uri = uri
        self.horizon = horizon
        self.start_date = start_date
        self.end_date = end_date
        self.model_info = {}
        self.preds = {}

    def initialize(self):
        print('-' * 100)
        print('initialize ...')
        try:
            qlib.init(provider_uri=self.provider_uri)
            print("QLib初始化成功!")

            # 加载实验记录
            exp_manager = MLflowExpManager(uri=self.uri, default_exp_name='default_exp')
            for h in self.horizon:
                exp_name = "{}{}".format('lightGBM_Alpha158_h', h)
                print(exp_name)
                self.model_info[exp_name] = {}
                exp = exp_manager.get_exp(experiment_name=exp_name)
                if not exp:
                    raise ValueError(f"找不到Name为{exp_name}的实验记录")

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
                    raise ValueError(f"实验{exp_name}中没有找到在线模型记录")

                # 选择最新的在线模型
                newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)
                # 加载模型
                model = newest_recorder.load_object("params.pkl")
                # 加载dataset
                dataset = newest_recorder.load_object("dataset")
                self.model_info[exp_name]['model'] = model
                self.model_info[exp_name]['dataset'] = dataset

                print(f"加载模型。实验Name: {exp_name}")

        except Exception as e:
            raise Exception(f"模型初始化失败: {str(e)}")

        assert len(self.horizon) == len(self.model_info)
        print(self.model_info)
        print('初始化成功 !')

    def predict(self, start_date, end_date):
        print('-' * 100)
        print('predict ...')
        for exp_name in self.model_info:
            print(exp_name)
            model = self.model_info[exp_name]['model']
            dataset = self.model_info[exp_name]['dataset']
            segments = {"test": (start_date, end_date)}
            dataset.config(
                handler_kwargs={"start_time": start_date, "end_time": end_date},
                segments=segments)
            # 准备数据
            dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
            # 预测
            df = model.predict(dataset)
            df = df.reset_index()
            df.columns = ['day', 'qlib_code', 'score']
            self.preds[exp_name] = df

    def merge_preds(self):
        print('-' * 100)
        print('merge_preds ...')
        merged = pd.DataFrame()
        for exp_name in self.preds:
            hr = int(exp_name.replace('lightGBM_Alpha158_h', ''))
            df = self.preds[exp_name]
            date = df['day'].iloc[0].strftime('%Y-%m-%d')
            next = pd.to_datetime(get_n_nexttrade_day(date, hr))
            df['day'] = next
            merged = pd.concat([merged, df], axis=0, ignore_index=True)
        return merged

    def write_to_mysql(self, merged: pd.DataFrame):
        print('-' * 100)
        print('write_to_mysql ...')
        data = []
        columns = merged.columns.to_list()
        for index, row in merged.iterrows():
            tmp = {}
            for c in columns:
                tmp[c] = row[c]
            data.append(tmp)
        print('data len: {}'.format(len(data)))

        conn = mysql_connect()
        columns_format = ["%({})s".format(c) for c in columns]
        sql = '''
            INSERT INTO monitor_return_rate({}) VALUES ({})
            ON DUPLICATE KEY UPDATE score=VALUES(score)
        '''.format(','.join(columns), ','.join(columns_format))

        with conn.cursor() as cursor:
            try:
                print(cursor.mogrify(sql, data[0]))
                cursor.executemany(sql, data)
                conn.commit()
            except Exception as e:
                conn.rollback()  # 回滚
                raise Exception('error in write to mysql: {}'.format(e))
            finally:
                conn.close()
        print('写入完成!!!')

    def main(self):
        try:
            # 初始化
            self.initialize()
            # 预测
            self.predict(self.start_date, self.end_date)
            print(self.preds)
            # 合并预测结果
            merged = self.merge_preds()
            # 写入mysql
            self.write_to_mysql(merged)
        except Exception as e:
            erro_info = traceback.format_exc()
            send_email('Strategy:lightGBM:predict', erro_info)

if __name__ == '__main__':
    t = time.time()
    fire.Fire(Predict)
    print('耗时：{}s'.format(round(time.time() - t, 4)))
    '''
        python predict.py main --start_date 2025-09-19 --end_date 2025-09-19 --horizon 1,2
    '''