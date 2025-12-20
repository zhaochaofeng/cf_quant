'''
    lightGBM 模型 迭代训练
'''

import os
import time
import traceback
from pprint import pprint

import fire
import pandas as pd
import qlib
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.model.trainer import TrainerR, Trainer
from qlib.model.trainer import _log_task_info, _exe_task
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.gen import RollingGen
from qlib.constant import REG_CN

from strategy.dataset import (
    prepare_data_config
)
from strategy.model import prepare_model_config
from utils import (
    LoggerFactory,
    MySQLDB,
    CStd, CMean,
    get_config
)


class LightGBMModelRolling:

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
                 uri: str = None,
                 instruments: str = 'csi300',
                 benchmark: str = 'SH000300',
                 exp_name: str = 'lightgbm_alpha_rolling',
                 is_online: bool = False,
                 is_mysql: bool = False,
                 trainer: Trainer = TrainerR,
                 rolling_step: int = 30,
                 ref: int = 2
                 ):
        self.provider_uri = provider_uri
        self.instruments = instruments
        self.benchmark = benchmark
        self.exp_name = exp_name
        self.is_online = is_online
        self.is_mysql = is_mysql
        self.trainer = trainer(train_func=self.task_train)
        self.rolling_step = rolling_step
        self.rolling_online_manager = None
        self.ref = ref   # 预测第 ref 天的收益率
        self.recorders = []

        self.logger = LoggerFactory.get_logger(__name__)
        if uri is None:
            uri = './mlruns'
        self.uri = uri
        self.init()

    def init(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'qlib init ...'))
        _setup_kwargs = {'custom_ops': [CStd, CMean]}
        config = get_config()
        qlib.init(
            default_conf="server",
            region=REG_CN,
            # expression_cache='DiskExpressionCache',
            redis_host=config['redis']['host'],
            redis_port=config['redis']['port'],
            redis_task_db=3,
            redis_password=config['redis']['password'],
            provider_uri=self.provider_uri,
            **_setup_kwargs)

        dataset = self.prepare_data()
        model = self.prepare_model()

        record = [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {
                    "dataset": "<DATASET>",
                    "model": "<MODEL>",
                },
            },
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
            }
        ]

        task = {
            "dataset": dataset,
            "model": model,
            "record": record
        }
        strategies = RollingStrategy(
            name_id=self.exp_name,
            task_template=task,
            rolling_gen=RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD)
        )

        self.rolling_online_manager = OnlineManager(strategies=strategies, trainer=self.trainer)

    def task_train(self, task_config: dict, experiment_name: str, recorder_name: str = None) -> Recorder:
        """
        重写 task_train 方法，获取 rids
        """
        with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
            _log_task_info(task_config)
            _exe_task(task_config)
            self.recorders.append(R.get_recorder().id)
            return R.get_recorder()

    def prepare_data(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'prepare_data ...'))
        segments = {
            'train': ('2021-03-11', '2024-06-28'),
            'valid': ('2024-07-01', '2024-11-27'),
            'test': ('2024-11-28', '2025-12-11')
        }

        # segments = {
        #     'train': ('2025-08-01', '2025-08-31'),
        #     'valid': ('2025-09-01', '2025-09-30'),
        #     'test': ('2025-10-01', '2025-10-31')
        # }
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ]
        infer_processors = [
            {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}},
            {
                "class": "Winsorize",
                "module_path": "utils.qlib_processor",
                "kwargs": {"fields_group": "feature", "k": 3}
            }
        ]
        kwargs = {
            'expand_feas': None,
            'is_win': False,
            'is_std': False,
            'ref': -self.ref
        }
        # 自定义因子
        factors = []
        factor_dic = {}
        for factor in factors:
            factor_dic.update(factor())
        fields, names = [], []
        for k, v in factor_dic.items():
            names.append(k)
            fields.append(v['exp'])
        kwargs.update({'expand_feas': (fields, names)})
        dataset = prepare_data_config(
            segments=segments,
            class_name='ExpAlpha158',
            module_path='strategy.dataset',
            instruments=self.instruments,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs
        )
        self.logger.info(dataset)
        return dataset

    def prepare_model(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'prepare_model ...'))
        kwargs = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            # "max_depth": 8,
            "max_depth": 3,
            "num_leaves": 210,
            "num_threads": 20,
            "early_stopping_rounds": 10  # 防止过拟合
        }
        model = prepare_model_config(class_name='LGBModel2', module_path='strategy.model', **kwargs)
        self.logger.info(model)
        return model

    def backtest(self, signals, metrics):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'backtest ...'))
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
                    "signal": signals.to_frame("score"),
                    "topk": 50,
                    "n_drop": 5,
                    "hold_thresh": (self.ref - 1)
                },
            }
        }

        report_normal, positions_normal = backtest_daily(
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max(),
            strategy=port_analysis_config['strategy'],
            executor=port_analysis_config['executor'],
            account=100000000,
            benchmark=self.benchmark,
            exchange_kwargs={
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }
        )

        analysis = dict()
        analysis["benchmark"] = risk_analysis(report_normal["bench"])
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)
        pprint(analysis_df)

        metrics.append(self.read_metrics(self.recorders,
                                         signals.index.get_level_values("datetime").max().strftime('%Y-%m-%d'),
                                         self.ref - 1))
        print('\n{}\n{}'.format('=' * 100, metrics))

    def read_metrics(self, rec_ids, day, hr):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'read_metrics ...'))
        metrics = {}
        df_ic, df_ric = [], []
        for rid in rec_ids:
            recorder = R.get_recorder(recorder_id=rid, experiment_name=self.exp_name)
            df_ic.append(recorder.load_object('sig_analysis/ic.pkl'))
            df_ric.append(recorder.load_object('sig_analysis/ric.pkl'))
        df_ic = pd.concat(df_ic, axis=0)
        df_ric = pd.concat(df_ric, axis=0)
        metrics['IC'] = df_ic.mean()
        metrics['ICIR'] = df_ic.mean() / df_ic.std()
        metrics['RIC'] = df_ric.mean()
        metrics['RICIR'] = df_ric.mean() / df_ric.std()
        metrics['day'] = day
        metrics['horizon'] = hr
        metrics['model'] = self.exp_name
        metrics['instruments'] = self.instruments
        return metrics

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

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'reset ...'))
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        try:
            t0 = time.time()
            self.reset()
            self.rolling_online_manager.first_train()
            signals = self.rolling_online_manager.prepare_signals()

            metrics = []
            self.backtest(signals, metrics)

            # 指标写入mysql
            if self.is_mysql:
                self.metrics_to_mysql(metrics)
            # OnlineManager 实例写入文件
            self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)
            self.logger.info('耗时：{}s'.format(round(time.time() - t0, 4)))

        except:
            err_msg = traceback.format_exc()
            self.logger.error(err_msg)
            # send_email('Strategy: lightgbm_alpha', err_msg)


if __name__ == '__main__':
    fire.Fire(LightGBMModelRolling)
