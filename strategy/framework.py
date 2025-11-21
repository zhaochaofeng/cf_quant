'''
    功能：模型训练框架
'''
import copy
from typing import List, Tuple

import time
import fire
import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from data.factor.factor_func import (
    MACD, BOLL, KDJ, WR, BIAS_Multi, CCI, ROC
)
from utils import (
    standardize, winsorize, CMean, CStd,
    get_trade_cal_inter, LoggerFactory
)
import torch
from qlib.contrib.model.pytorch_transformer import TransformerModel

class TransformerModel2(TransformerModel):
    def __init__(self, GPU=0, d_feat=157, seed=0):
        super().__init__(GPU=GPU, d_feat=d_feat, seed=seed)
        self.device = self.get_device(GPU)
        self.model.to(self.device)

    def get_device(self, GPU=0, return_str=False):
        """
        Get the appropriate device (CUDA, MPS, or CPU) based on availability.
        Parameters
        ----------
        GPU : int
            the GPU ID used for training. If >= 0 and CUDA is available, use CUDA.
        return_str : bool
            if True, return device as string; if False, return torch.device object.
        Returns
        -------
        torch.device or str
            The device to use for computation.
        """
        USE_CUDA = torch.cuda.is_available() and GPU >= 0
        USE_MPS = torch.backends.mps.is_available()

        # Default to CPU, then check for GPU availability
        device_str = 'cpu'
        if USE_CUDA:
            device_str = f'cuda:{GPU}'
        elif USE_MPS:
            device_str = 'mps'

        if return_str:
            return device_str
        else:
            return torch.device(device_str)

class ExpAlpha158(Alpha158):
    def __init__(self,
                 use_expand_feas: bool = False,
                 is_win: bool = False,
                 is_std: bool = False,
                 **kwargs
                 ):
        """
        Args:
            use_expand_feas: 是否使用扩展特征
            is_win: 是否取极值
            is_std: 是否标准化
        """
        self.use_expand_feas = use_expand_feas
        self.is_win = is_win
        self.is_std = is_std
        super().__init__(**kwargs)

    def get_feature_config(self):
        # 获取原始Alpha158特征配置
        fields, names = super().get_feature_config()
        # 添加自定义特征
        if self.use_expand_feas:
            factors = [MACD(), BOLL(), KDJ(), WR(), BIAS_Multi(), CCI(), ROC()]
            for factor in factors:
                for name, info in factor.items():
                    fields.append(info["exp"])
                    names.append(name)
        if self.is_win:
            fields = [winsorize(f, 3) for f in fields]
        if self.is_std:
            fields = [standardize(f) for f in fields]
        # 删除 VWAP0 列
        # idx = names.index("VWAP0")
        # fields.pop(idx)
        # names.pop(idx)
        return fields, names


class LGBModel2(LGBModel):
    def _prepare_data_finetune(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        """
        Prepare data for finetune
        """
        ds_l = []
        assert "train" in dataset.segments
        # for key in ["train", ["valid", "test"]]:
        for key in [["valid", "test"]]:
            # if key in dataset.segments:
            df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            if isinstance(df, list):  # 合并 valid 和 test
                df = pd.concat(df, axis=0, join="outer")
            if df.empty:
                raise ValueError("Empty data from dataset, please check your dataset config.")
            x, y = df["feature"], df["label"]

            # Lightgbm need 1D array as its label
            if y.values.ndim == 2 and y.values.shape[1] == 1:
                y = np.squeeze(y.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            if reweighter is None:
                w = None
            elif isinstance(reweighter, Reweighter):
                w = reweighter.reweight(df)
            else:
                raise ValueError("Unsupported reweighter type.")
            ds_l.append((lgb.Dataset(x.values, label=y, weight=w), key))
        return ds_l

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        dtrain, valid_test = self._prepare_data_finetune(dataset, reweighter)  # pylint: disable=W0632

        valid_test = valid_test[0]
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            valid_test,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[valid_test],
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
        )


class Model:
    def __init__(self,
                 segments: dict[str, tuple[str, str]] = {
        'train': ('2015-01-05', '2023-12-31'),
        # 'train': ('2023-01-05', '2023-12-31'),
        'valid': ('2024-01-01', '2024-12-31'),
        'test': ('2025-01-01', '2025-11-15')
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
                 date_type: str = 'ds'  # 数据类型 DatasetH/TSDatasetH
                 ):
        self.segments = segments
        self.market = market
        self.benchmark = benchmark
        self.experiment_name = experiment_name
        self.use_expand_feas = use_expand_feas
        self.is_win = is_win
        self.is_std = is_std
        self.step = step
        self.date_type = date_type
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
                self.logger.info('data round: {}'.format(i+1))
                start = train_list[i]
                end = train_list[min(i + step, len(train_list)) - 1]
                self.logger.info('train: ({}, {})'.format(start, end))
                dataset = self.prepare_dataset(start, end, self.date_type, ExpAlpha158, **kwargs)
                self.logger.info('dateset: {}'.format(dataset))

                self.model.fit(dataset)
            R.save_objects(**{'params.pkl': self.model})
            R.save_objects(**{'dataset': dataset})  # 保存最后迭代的dateset配置

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
            model = TransformerModel2()
            self.logger.info('device: {}'.format(model.device))
        self.logger.info('model: {}'.format(model))
        return model

    def prepare_dataset(self, start, end, date_type, handler_model, **kwargs):
        """
            构建训练数据
        Parameters
        ----------
        start: train_start
        end：train_end
        date_type: 数据类型。ds: DatasetH, ts: TSDatasetH
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

        if date_type == 'ts':
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
                   use_expand_feas=False,
                   is_win=False,
                   is_std=False
                   )
        self.logger.info('耗时：{}s'.format(round(time.time()-t, 4)))

if __name__ == '__main__':
    fire.Fire(Model)
