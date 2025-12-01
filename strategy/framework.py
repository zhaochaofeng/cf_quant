'''
    功能：模型训练框架
'''
import copy
import time

import fire
import qlib
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from strategy.dataset import ExpAlpha158
from qlib.contrib.data.handler import Alpha360
from strategy.model import LGBModel2, TransformerModel2
from utils import (
    CMean, CStd,
    get_trade_cal_inter, LoggerFactory
)


class Model:
    def __init__(self,
                 segments: dict[str, tuple[str, str]] = {
        'train': ('2015-01-05', '2023-12-31'),
        'valid': ('2024-01-01', '2024-12-31'),
        'test': ('2025-01-01', '2025-11-15')
        #  "train": ("2008-01-01", "2014-12-31"),
        #  "valid": ("2015-01-01", "2016-12-31"),
        #  "test": ("2017-01-01", "2020-08-01")
},
                 market: str = 'csi300',
                 benchmark: str = "SH000300",
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
                 experiment_name: str = 'workflow',
                 is_finetune: bool = False,
                 use_expand_feas: bool = False,  # 是否使用自定义特征
                 is_win: bool = True,
                 is_std: bool = False,
                 step: int = 3000,
                 model_name: str = 'transformer',
                 data_type: str = 'ds'  # 数据类型 DatasetH/TSDatasetH
                 ):
        self.segments = segments
        self.market = market
        self.benchmark = benchmark
        self.experiment_name = experiment_name
        self.use_expand_feas = use_expand_feas
        self.is_win = is_win
        self.is_std = is_std
        self.step = step
        self.data_type = data_type
        self.model_name = model_name

        _setup_kwargs = {'custom_ops': [CMean, CStd]}  # 注册自定义操作
        qlib.init(provider_uri=provider_uri, **_setup_kwargs)
        self.logger = LoggerFactory.get_logger(__name__)
        self.model = self.load_model(is_finetune, model_name)

    def train(self, step: int = 3000, is_online: bool = False, **kwargs):
        """
        Parameters
        ----------
        step: 训练的步长(day)
        is_online: 是否使用 valid,test 数据训练，生成线上模型
        """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'train ...'))
        train_start, train_end = self.segments['train']
        self.logger.info('total train: ({}, {})'.format(train_start, train_end))
        train_list = get_trade_cal_inter(train_start, train_end)
        with R.start(experiment_name=self.experiment_name):
            dataset = None
            for i in range(0, len(train_list), step):
                self.logger.info('data_new round: {}'.format(i+1))
                start = train_list[i]
                end = train_list[min(i + step, len(train_list)) - 1]
                self.logger.info('train: ({}, {})'.format(start, end))
                dataset = self.prepare_dataset(start, end, self.data_type, Alpha360, **kwargs)
                self.logger.info('dataset: {}'.format(dataset))

                self.model.fit(dataset)
            R.save_objects(**{'params.pkl': self.model})
            R.save_objects(**{'dataset': dataset})  # 保存最后迭代的dataset配置

            recorder = R.get_recorder()
            self.backtest(self.model, dataset, recorder)
        if is_online:
            self.logger.info('online train ...')
            with R.start(experiment_name='{}_online'.format(self.experiment_name)):
                self.model.finetune(dataset, num_boost_round=20)
                R.save_objects(**{'params.pkl': self.model})
                R.save_objects(**{'dataset': dataset})

    def load_model(self, is_finetune: bool = False, model_name: str = 'gbm'):
        """
        Parameters
        ----------
        is_finetune
        model_name: gbm / transformer
        """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'load_model ...'))
        if is_finetune:
            self.logger.info('load model from newest recorder ...')
            exp = R.get_exp(experiment_name=self.experiment_name)
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
                raise ValueError(f"实验{self.experiment_name}中没有找到在线模型记录")
            # 选择最新的在线模型
            newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)
            recorder = newest_recorder
            # 加载模型
            model = recorder.load_object("params.pkl")
        elif model_name == 'gbm':
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
        else:
            self.logger.info('create transformer model object ...')
            model = TransformerModel2(d_feat=359)
            self.logger.info('device: {}'.format(model.device))
        self.logger.info('model: {}'.format(model))
        return model

    def prepare_dataset(self, start, end, data_type, handler_model, **kwargs):
        """
            构建训练数据
        Parameters
        ----------
        start: train_start
        end：train_end
        data_type: 数据类型。ds: DatasetH, ts: TSDatasetH
        handler_model: 特征处理类

        Returns: DatasetH
        -------
        """
        self.logger.info('\n{}\n{}'.format('-' * 50, 'prepare_dataset ...'))

        segments = copy.deepcopy(self.segments)
        segments['train'] = (start, end)

        if self.model_name == 'gbm':
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},

            ]
            infer_processors = [
                {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}}
            ]
        else:
            learn_processors = [
               {"class": "DropnaLabel"},
               {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ]
            infer_processors = [
                {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}},
                {"class": "ProcessInf", "kwargs": {}},
                {"class": "ZScoreNorm", "kwargs": {}},
                {"class": "Fillna", "kwargs": {}},
            ]

        data_handler_config = {
            "start_time": segments['train'][0],
            "end_time": segments['test'][1],
            "fit_start_time": start,
            "fit_end_time": end,
            "instruments": self.market,
            "learn_processors": learn_processors,
            "infer_processors": infer_processors
        }

        data_handler_config.update(kwargs)
        handler = handler_model(**data_handler_config)

        if data_type == 'ts':
            dataset = {
                "class": "TSDatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler,
                    "segments": segments,
                    "step_len": 20
                }
            }
        else:
            dataset = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler,
                    "segments": segments
                },
            }

        return init_instance_by_config(dataset)

    def backtest(self, model, dataset, recorder) -> None:
        self.logger.info('\n{}\n{}'.format('=' * 100, 'backtest ...'))
        test = self.segments['test']

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
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "model": model,
                    "dataset": dataset,
                    "topk": 50,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": test[0],
                "end_time": test[1],
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    # "open_cost": 0.0003,
                    # "close_cost": 0.0013,
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        # 预测信号
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        sar = SigAnaRecord(recorder, ana_long_short=False)
        sar.generate()

        # 回测
        par = PortAnaRecord(recorder, config=port_analysis_config, risk_analysis_freq='day')
        par.generate()

    def main(self):
        t = time.time()
        self.train(step=3000,
                   is_online=False,
                   # use_expand_feas=False,
                   # is_win=False,
                   # is_std=False
                   )
        self.logger.info('耗时：{}s'.format(round(time.time()-t, 4)))

if __name__ == '__main__':
    fire.Fire(Model)
