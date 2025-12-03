'''
    lightGBM 模型训练
'''

import fire
import copy
import time
import traceback
import pandas as pd
from datetime import datetime
import qlib
from qlib.workflow import R
from qlib.data import D
from qlib.utils.data import zscore
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from utils import (
    LoggerFactory,
    MySQLDB,
    get_n_pretrade_day,
    send_email
)
from strategy.dataset import (
    prepare_data,
    ExpAlpha158,
    MultiHorizonGen,
    dataframe_to_dataset
)
from strategy.model import LGBModel2


class LightGBMModel:
    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/cn_data',
                 uri: str = None,
                 segments: dict = None,
                 instruments: str = 'csi300',
                 benchmark: str = 'SH000300',
                 exp_name: str = 'lightgbm_alpha',
                 is_finetune: bool = False,
                 is_online: bool = False,
                 is_mysql: bool = True,
                 horizon: list = None,
                 start_wid: int = 1,
                 test_wid: int = 200,
                 valid_wid: int = 100,
                 train_wid: int = 500,
                 ):
        self.provider_uri = provider_uri
        self.instruments = instruments
        self.benchmark = benchmark
        self.exp_name = exp_name
        self.is_finetune = is_finetune
        self.is_online = is_online
        self.is_mysql = is_mysql
        self.start_wid = start_wid
        self.test_wid = test_wid
        self.valid_wid = valid_wid
        self.train_wid = train_wid

        self.logger = LoggerFactory.get_logger(__name__)
        if horizon is None:
            horizon = [1]
        self.horizon = horizon
        if segments is not None:
            self.segments = segments
        else:
            self.segments = self.date_interval()
        if uri is None:
            uri = './mlruns'
        self.uri = uri
        self.init()

    def init(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'qlib init ...'))
        qlib.init(provider_uri=self.provider_uri)

    def date_interval(self) -> dict:
        ''' 训练 / 验证 / 测试 时间区间'''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'date_interval ...'))
        now = datetime.now().strftime('%Y-%m-%d')
        test_end = get_n_pretrade_day(now, self.start_wid)
        test_start = get_n_pretrade_day(test_end, self.test_wid)
        valid_end = get_n_pretrade_day(test_start, 1)
        valid_start = get_n_pretrade_day(valid_end, self.valid_wid)
        train_end = get_n_pretrade_day(valid_start, 1)
        train_start = get_n_pretrade_day(train_end, self.train_wid)

        train_inter = (train_start, train_end)
        valid_inter = (valid_start, valid_end)
        test_inter = (test_start, test_end)

        segments = {
            "train": train_inter,
            "valid": valid_inter,
            "test": test_inter,
        }
        self.logger.info(segments)
        return segments

    def prepare_data(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'prepare_data ...'))
        learn_processors = [
            # {"class": "DropnaLabel"},
            # {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ]
        infer_processors = [
            {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}}
        ]
        kwargs = {
            'expand_feas': None,
            'is_win': False,
            'is_std': False,
            'ref': -2
        }
        dataset, config = prepare_data(self.segments,
                               handler_model=ExpAlpha158,
                               instruments=self.instruments,
                               learn_processors=learn_processors,
                               infer_processors=infer_processors,
                               **kwargs
                               )
        return dataset, config

    def get_label(self, start_time, end_time, hr, instruments):
        col_name = f'LABEL{hr}'
        fields = [f"Ref($close, -({hr}+1)) / Ref($close, -1) - 1"]
        instruments = D.instruments(market=instruments)
        df = D.features(instruments=instruments, fields=fields, start_time=start_time, end_time=end_time)
        df.columns = pd.MultiIndex.from_tuples([('label', col_name)])

        df_learn = copy.deepcopy(df)   # 用于训练(learn)
        df_learn.dropna(inplace=True, axis=0)
        # 训练的标签 CSZScoreNorm
        df_learn = df_learn.groupby('datetime', group_keys=False).apply(zscore)
        df_learn = df_learn.swaplevel()   # 索引转化为 <datetime, instrument>
        df_learn.sort_index(inplace=True)

        df_infer = df.swaplevel()     # 用于回测
        df_infer.sort_index(inplace=True)
        return df_learn, df_infer

    def load_model(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'load_model ...'))
        if self.is_finetune:
            self.logger.info('load model from newest recorder ...')
            exp = R.get_exp(experiment_name=self.exp_name)
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
                raise ValueError(f"实验{self.exp_name}中没有找到在线模型记录")
            # 选择最新的在线模型
            newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)
            recorder = newest_recorder
            # 加载模型
            model = recorder.load_object("params.pkl")
        else:
            self.logger.info('create gbm model object ...')
            kwargs = {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            }
            model = LGBModel2(**kwargs)
        return model

    def backtest(self, dataset, model, recorder, hr, metrics):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'backtest ...'))
        segments = dataset.segments
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "model": model,
                    "dataset": dataset,
                    "topk": 50,
                    "n_drop": 5,
                    "hold_thresh": hr
                },
            },
            "backtest": {
                "start_time": segments['test'][0],
                "end_time": segments['test'][1],
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, config=port_analysis_config, risk_analysis_freq='day')
        par.generate()

        metrics.append(self.read_metrics(recorder, segments['test'][1], hr))

    def read_metrics(self, recorder, day, hr):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'read_metrics ...'))
        path = '{}/{}/{}/metrics'.format(recorder.uri.replace('file:', ''), recorder.experiment_id, recorder.id)
        print(path)
        metrics = {}
        map = {'IC': 'IC', 'ICIR': 'ICIR', 'Rank IC': 'RIC', 'Rank ICIR': 'RICIR'}
        for m in map.keys():
            metrics[map[m]] = pd.read_csv(path + '/' + m, sep=' ', header=None).iloc[0, 1]
        metrics['day'] = day
        metrics['horizon'] = hr
        metrics['model'] = self.exp_name
        metrics['instruments'] = self.instruments
        return metrics

    def print_info(self, fea_learn, label_learn, data_learn, dataset_learn,
                   fea_infer, label_infer, data_infer, dataset_infer
                   ):
        self.logger.info('dataset_learn: {}'.format('-' * 50))
        self.logger.info("features_learn shape: {}".format(fea_learn.shape))
        self.logger.info("label_learn shape: {}".format(label_learn.shape))
        self.logger.info("data_learn shape: {}".format(data_learn.shape))
        # self.logger.info(dataset_learn.prepare(col_set=['feature', 'label'], segments='train'))
        self.logger.info('dataset_infer: {}'.format('-' * 50))
        self.logger.info("features_infer shape: {}".format(fea_infer.shape))
        self.logger.info("label_infer shape: {}".format(label_infer.shape))
        self.logger.info("data_infer shape: {}".format(data_infer.shape))
        # self.logger.info(dataset_infer.prepare(col_set=['feature', 'label'], segments='train'))

    def train(self, dataset_learn, dataset_infer, model, hr, metrics, ds_config):
        exp_name = f"{self.exp_name}_h{hr}"
        with R.start(experiment_name=exp_name, uri=self.uri):
            model.fit(dataset_learn)
            R.save_objects(**{'params.pkl': model})
            R.save_objects(**{'dataset': ds_config})
            recorder = R.get_recorder()
            self.backtest(dataset_infer, model, recorder, hr, metrics)

        if self.is_online:
            self.logger.info('\n{}\n{}'.format('-' * 50, 'online training ...'))
            with R.start(experiment_name='online_{}'.format(exp_name), uri=self.uri):
                model.finetune(dataset_learn, num_boost_round=10, verbose_eval=10)
                R.save_objects(**{'params.pkl': model})
                R.save_objects(**{'dataset': ds_config})

    def metrics_to_mysql(self, metrics: list) -> None:
        self.logger.info('\n{}\n{}'.format('=' * 100, 'metrics_to_mysql ...'))
        if len(metrics) == 0:
            raise ValueError('metrics is empty')
        feas = list(metrics[0].keys())
        feas_format = ['%({})s'.format(f) for f in feas]
        with MySQLDB() as db:
            sql = '''INSERT INTO monitor_model_metrics ({}) VALUES ({})'''.format(','.join(feas), ','.join(feas_format))
            db.executemany(sql, metrics)

    def main(self):
        try:
            t0 = time.time()
            dataset, ds_config = self.prepare_data()
            task = {
                "dataset": ds_config,
            }

            fea_learn = pd.concat(
                dataset.prepare(segments=['train', 'valid', 'test'], col_set=['feature'], data_key='learn'), axis=0)
            fea_infer = pd.concat(
                dataset.prepare(segments=['train', 'valid', 'test'], col_set=['feature'], data_key='infer'), axis=0)

            # 生成多周期tasks
            multi_gen = MultiHorizonGen(horizon=self.horizon, label_leak_n=2)
            tasks = multi_gen.generate(task)
            assert len(tasks) == len(self.horizon)
            metrics = []
            for i, t in enumerate(tasks):
                hr = t['extra']['horizon']
                self.logger.info('\n{}\n{}: {}'.format('*' * 100, 'horizon', hr))
                model = self.load_model()
                label_learn, label_infer = self.get_label(self.segments['train'][0],
                                                          self.segments['test'][1],
                                                          hr,
                                                          self.instruments)

                data_learn = pd.concat([fea_learn, label_learn], axis=1, join='inner')
                data_infer = pd.concat([fea_infer, label_infer], axis=1, join='inner')

                segments = t["dataset"]["kwargs"]["segments"]
                dataset_learn = dataframe_to_dataset(data_learn, segments)
                dataset_infer = dataframe_to_dataset(data_infer, segments)

                self.print_info(
                    fea_learn, label_learn, data_learn, dataset_learn,
                    fea_infer, label_infer, data_infer, dataset_infer)

                self.train(dataset_learn, dataset_infer, model, hr, metrics, dataset)

            # 指标写入mysql
            if self.is_mysql:
                self.metrics_to_mysql(metrics)
            self.logger.info('耗时：{}s'.format(round(time.time() - t0, 4)))
        except:
            err_msg = traceback.format_exc()
            self.logger.error(err_msg)
            send_email('Strategy: lightgbm_alpha', err_msg)


if __name__ == '__main__':
    fire.Fire(LightGBMModel)
