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


class MasterModelRolling:

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
                 uri: str = None,
                 instruments: str = 'csi300',
                 benchmark: str = 'SH000300',
                 exp_name: str = 'master_alpha_rolling',
                 is_online: bool = False,
                 is_mysql: bool = False,
                 trainer: Trainer = TrainerR,
                 rolling_step: int = 30,
                 ref: int = 5,
                 step_len: int = 8,
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
        self.step_len = step_len
        self.recorders = []

        self.logger = LoggerFactory.get_logger(__name__)
        if uri is None:
            uri = './mlruns'
        self.uri = uri
        self.init()

    def init(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'qlib init ...'))
        config = get_config()
        qlib.init(
            default_conf="server",
            region=REG_CN,
            redis_host=config['redis']['host'],
            redis_port=config['redis']['port'],
            redis_task_db=3,
            redis_password=config['redis']['password'],
            provider_uri=self.provider_uri
        )

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
            rolling_gen=RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD, trunc_days=self.ref+self.step_len)
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
            'train': ('2015-01-01', '2024-05-31'),
            'valid': ('2024-06-01', '2024-12-31'),
            'test': ('2025-01-01', '2025-12-01')
        }

        # segments = {
        #     'train': ('2008-02-01', '2020-03-31'),
        #     'valid': ('2020-04-01', '2020-06-30'),
        #     'test': ('2020-07-01', '2022-12-31')
        # }
        # segments = {
        #     'train': ('2008-01-01', '2020-03-31'),
        #     'valid': ('2020-04-10', '2020-06-30'),
        #     'test': ('2020-07-10', '2022-12-31')
        # }

        # segments = {
        #     'train': ('2018-08-01', '2025-08-31'),
        #     'valid': ('2025-09-10', '2025-10-10'),
        #     'test': ('2025-10-20', '2025-11-20')
        # }
        learn_processors = [
            {"class": "DropnaLabel"},
            # {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
        ]
        infer_processors = [
            # {"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            # {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},  # test_epoch中需要使用，所以放在infer_processors中
        ]
        kwargs = {
            'ref': -self.ref
        }
        # 自定义因子
        factors = []
        if len(factors) > 0:
            factor_dic = {}
            for factor in factors:
                factor_dic.update(factor())
            fields, names = [], []
            for k, v in factor_dic.items():
                names.append(k)
                fields.append(v['exp'])
            fields_all, names_all = kwargs.get('expand_feas', ([], []))
            fields_all.extend(fields)
            names_all.extend(names)
            kwargs.update(expand_feas=(fields_all, names_all))

        # 市场因子
        fields, names = [], []
        for s in ['SH000300', 'SH000905', 'SH000906']:
            f = """Mask($close/Ref($close,1)-1, "{}")""".format(s)
            fields.append(f)
            names.append(f)
            for d in [5, 10, 20, 30, 60]:
                mean_close = """ Mask(Mean($close/Ref($close,1)-1,{}), "{}") """.format(d, s)
                std_close = """ Mask(Std($close/Ref($close,1)-1,{}), "{}") """.format(d, s)

                mean_volume = """Mask(Mean($amount,{})/$amount, "{}")""".format(d, s)
                std_volume = """Mask(Std($amount,{})/$amount, "{}")""".format(d, s)

                fields.extend([mean_close, std_close, mean_volume, std_volume])
                names.extend([mean_close, std_close, mean_volume, std_volume])
        fields_all, names_all = kwargs.get('expand_feas', ([], []))
        fields_all.extend(fields)
        names_all.extend(names)
        kwargs.update(expand_feas=(fields_all, names_all))

        dataset = prepare_data_config(
            segments=segments,
            class_name='ExpAlpha158',
            module_path='strategy.dataset',
            instruments=self.instruments,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            data_type='ts',
            step_len=self.step_len,
            **kwargs
        )
        '''
        import pickle
        from qlib.utils import init_instance_by_config
        data = init_instance_by_config(dataset)

        train = data.prepare(segments='train', col_set=['feature', 'label'], data_key='learn')
        valid = data.prepare(segments='valid', col_set=['feature', 'label'], data_key='learn')
        test = data.prepare(segments='test', col_set=['feature', 'label'], data_key='infer')

        train.config(**{'fillna_type': 'ffill+bfill'})
        valid.config(**{'fillna_type': 'ffill+bfill'})
        test.config(**{'fillna_type': 'ffill+bfill'})

        base_path = '/Users/chaofeng/code/MASTER_reader/data/custom_data_qlib'
        os.makedirs(base_path, exist_ok=True)
        pickle.dump(train, open(os.path.join(base_path, 'csi300_dl_train.pkl'), 'wb'))
        pickle.dump(valid, open(os.path.join(base_path, 'csi300_dl_valid.pkl'), 'wb'))
        pickle.dump(test, open(os.path.join(base_path, 'csi300_dl_test.pkl'), 'wb'))
        '''

        self.logger.info(dataset)
        return dataset

    def prepare_model(self):
        self.logger.info('\n{}\n{}'.format('=' * 100, 'prepare_model ...'))
        kwargs = {
            "d_feat": 158,
            "d_model": 256,
            "t_nhead": 4,
            "s_nhead": 2,
            "n_epochs": 30,
            "lr": 1e-5,
            "GPU": 0,
            "seed": 0,
            "beta": 5,
            "train_stop_loss_thred": 0.97,
        }
        model = prepare_model_config(class_name='MASTERModel',
                                     module_path='strategy.master.pytorch_master_ts',
                                     **kwargs)
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
    fire.Fire(MasterModelRolling)
