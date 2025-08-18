"""
    训练和定时更新模型
"""

import os
import fire
import qlib
from qlib.model.trainer import TrainerR, TrainerRM
from qlib.workflow import R
from qlib.data import D
from qlib.data.filter import NameDFilter
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.online.manager import OnlineManager

from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

import pandas as pd
from pprint import pprint
from datetime import datetime, timedelta

from utils.utils import sql_engine
from utils.utils import tushare_pro
pro = tushare_pro()


class RollingOnlineTrain:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/custom_data_hfq",
        region="cn",
        trainer=TrainerR(),
        rolling_step=10,
        tasks=None,
        add_tasks=None,
        market='all'
    ):
        qlib.init(provider_uri=provider_uri, region=region)
        self.market = market

        if tasks is None:
            instruments = self.choose_stocks()
            data_handler_config = {
                "start_time": "2023-08-02",
                "end_time": "2025-08-01",
                "fit_start_time": "2023-08-02",
                "fit_end_time": "2025-04-01",
                "instruments": instruments,
                # "instruments": 'csi300',
            }
            # 模型和数据的配置参数
            task = {
                # "model": {
                #     "class": "LGBModel",
                #     "module_path": "qlib.contrib.model.gbdt",
                #     "kwargs": {
                #         "loss": "mse",
                #         "colsample_bytree": 0.8879,
                #         "learning_rate": 0.0421,
                #         "subsample": 0.8789,
                #         "lambda_l1": 205.6999,
                #         "lambda_l2": 580.9768,
                #         "max_depth": 8,
                #         "num_leaves": 210,
                #         "num_threads": 20,
                #     },
                # },
                'model': {'class': 'LGBModel', 'module_path': 'qlib.contrib.model.gbdt'},
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158",
                            "module_path": "qlib.contrib.data.handler",
                            "kwargs": data_handler_config,
                        },
                        "segments": {
                            "train": ("2023-08-02", "2025-01-01"),
                            "valid": ("2025-01-02", "2025-04-01"),
                            "test": ("2025-04-02", "2025-06-01"),
                        },
                    },
                },
                'record': [
                    {
                        'class': 'SignalRecord',
                        'module_path': 'qlib.workflow.record_temp',
                        'kwargs': {
                            'dataset': '<DATASET>',
                            'model': '<MODEL>'
                        },
                    },
                    {
                        'class': 'SigAnaRecord',
                        'module_path': 'qlib.workflow.record_temp'
                    }
                ]
            }
            tasks = [task]

        self.tasks = tasks
        self.add_tasks = add_tasks
        self.rolling_step = rolling_step
        strategies = []
        for task in tasks:
            name_id = task["model"]["class"]  # NOTE: Assumption: The model class can specify only one strategy
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None),
                )
            )
        self.trainer = trainer
        self.rolling_online_manager = OnlineManager(strategies, trainer=self.trainer)

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    '''
    def get_st_stocks(self):
        
        #获取30天之内为ST/*ST的股票
        
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            today = datetime.now().strftime('%Y%m%d')
            engine = sql_engine()
            sql = """ select ts_code, name from cf_quant.stock_info where day>={} and day<={}""".format(start_date, today)
            print(sql)
            stocks_info = pd.read_sql(sql, engine)
            st_stocks = stocks_info[
                (stocks_info['name'].str.contains('ST')) |
                (stocks_info['name'].str.contains('退'))
                ]
            return set(st_stocks['ts_code'].tolist())
        except Exception as e:
            raise Exception("Get ST stock info failed: {}".format(e))
    '''

    def choose_stocks(self):
        print('choose_stocks ...')
        ''' 股票筛选  '''
        # 主板。第3-4位数字为60或00
        nameDFilter = NameDFilter(name_rule_re='^[A-Za-z]{2}(60|00)')
        instruments = D.instruments(
            market=self.market,
            filter_pipe=[nameDFilter]
        )
        stocks = D.list_instruments(instruments, as_list=True)
        print('主板总股票数：{}'.format(len(stocks)))

        # 获取所有股票基本信息
        engine = sql_engine()
        # stocks_info = pro.stock_basic()
        start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        target_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        print('start_date: {}, today: {}, target_day: {}'.format(start_date, today, target_date))
        sql = """select ts_code, name, list_date from cf_quant.stock_info where day>='{}' and day<='{}'""".format(start_date, today)
        stocks_info = pd.read_sql(sql, engine)
        # 过滤ST、退市和次新股
        stocks_filter = stocks_info[
                (stocks_info['ts_code'].str.contains('ST')) |   # ST股票
                (stocks_info['name'].str.contains('退')) |      # 非退市股票
                (stocks_info['list_date'] > target_date)        # 非次新股
            ]
        print('stocks_filter len: {}'.format(len(stocks_filter)))
        # 转换股票代码格式以匹配qlib格式
        stocks_filter = stocks_filter['ts_code'].apply(lambda x: '{}{}'.format(x[7:9], x[0:6])).tolist()
        stocks = list(set(stocks) - set(stocks_filter))
        print('过滤后股票数：{}'.format(len(stocks)))
        return stocks

    def worker(self):
        # train tasks by other progress or machines for multiprocessing
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            for task in self.tasks + self.add_tasks:
                name_id = task["model"]["class"]
                self.trainer.worker(experiment_name=name_id)
        else:
            print(f"{type(self.trainer)} is not supported for worker.")

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        for task in self.tasks:
            name_id = task["model"]["class"]
            exp = R.get_exp(experiment_name=name_id)
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        print("========== reset ==========")
        self.reset()
        print("========== first_run ==========")
        self.rolling_online_manager.first_train()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def routine(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== routine ==========")
        self.rolling_online_manager.routine()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        signals = self.rolling_online_manager.get_signals()
        print(signals)
        print("========== backtest ==========")
        self.backtest(signals)
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def backtest(self, signals):
        STRATEGY_CONFIG = {
            "topk": 50,
            "n_drop": 5,
            "signal": signals.to_frame("score"),
        }
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max() - pd.Timedelta(days=1),
            strategy=strategy_obj,
        )
        print(report_normal)

        analysis = dict()
        analysis["bench"] = risk_analysis(report_normal["bench"])
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)

    def main(self):
        self.first_run()
        self.routine()

if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python rolling_online_management.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python rolling_online_management.py routine

    ####### to define your own parameters, use `--`
    # python rolling_online_management.py first_run --exp_name='your_exp_name' --rolling_step=40
    fire.Fire(RollingOnlineTrain)
