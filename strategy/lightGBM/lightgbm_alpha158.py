'''
    功能：lightGBM_Alpha158策略的模型训练
'''
import fire
import pandas as pd
import qlib
from qlib.data import D
from qlib.workflow import R
from qlib.data.filter import NameDFilter
from qlib.utils import flatten_dict
from qlib.utils import init_instance_by_config
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from datetime import datetime, timedelta
from utils.utils import sql_engine
from utils.utils import get_n_pretrade_day

class LightGBMAlpha158:
    def __init__(self,
        market='all',
        benchmark="SH000300",
        provider_uri='~/.qlib/qlib_data/custom_data_hfq',
        experiment_name='lightGBM_Alpha158',
        start_wid=1,      # test_end 向前移动的天数。至少前移1天，保证回测时不出错
        test_wid=100,     # 测试集时间宽度
        valid_wid=100,    # 验证集时间宽度
        train_wid=500    # 训练集时间宽度
    ):
        self.market = market
        self.benchmark = benchmark
        self.experiment_name = experiment_name
        self.start_wid = start_wid
        self.test_wid = test_wid
        self.valid_wid = valid_wid
        self.train_wid = train_wid
        qlib.init(provider_uri=provider_uri)

    def choose_stocks(self, start_time, end_time):
        ''' 股票筛选  '''
        print('choose_stocks ...')
        # 主板。第3-4位数字为60或00
        nameDFilter = NameDFilter(name_rule_re='^[A-Za-z]{2}(60|00)')
        instruments = D.instruments(
            market=self.market,
            filter_pipe=[nameDFilter]
        )

        print('start_time: {}, end_time: {}'.format(start_time, end_time))
        stocks = D.list_instruments(instruments, as_list=True, start_time=start_time, end_time=end_time)
        print('主板总股票数：{}'.format(len(stocks)))

        # 15天之内的 ST股、次新股
        engine = sql_engine()
        dt1 = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
        dt2 = datetime.now().strftime('%Y-%m-%d')
        target_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        print('dt1: {}, dt2: {}, target_day: {}'.format(dt1, dt2, target_date))
        sql = """select ts_code, name, list_date from cf_quant.stock_info where day>='{}' and day<='{}'""".format(dt1, dt2)
        stocks_info = pd.read_sql(sql, engine)
        # 过滤ST、退市和次新股
        stocks_filter = stocks_info[
                (stocks_info['name'].str.contains('ST')) |      # ST股票
                (stocks_info['name'].str.contains('退')) |      # 退市股票
                (stocks_info['list_date'] > target_date)        # 次新股
            ]
        # 转换股票代码格式以匹配qlib格式
        stocks_filter = stocks_filter['ts_code'].apply(lambda x: '{}{}'.format(x[7:9], x[0:6])).unique().tolist()
        print('stocks_filter len: {}'.format(len(stocks_filter)))
        stocks = set(stocks) - set(stocks_filter)
        # 排除非benchmark指数
        index_list = ['SH000905', 'SH000903']
        stocks = list(stocks - set(index_list))
        print('过滤后股票数：{}'.format(len(stocks)))
        print(stocks[0:10])
        return stocks

    def date_interval(self):
        ''' 训练 / 验证 / 测试 时间区间'''
        print('date_interval ...')
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

        print('train_inter: {}'.format(train_inter))
        print('valid_inter: {}'.format(valid_inter))
        print('test_inter: {}'.format(test_inter))

        return train_inter, valid_inter, test_inter

    def main(self):
        train_inter, valid_inter, test_inter = self.date_interval()

        instruments = self.choose_stocks(train_inter[0], test_inter[1])

        data_handler_config = {
            "start_time": train_inter[0],
            "end_time": test_inter[1],
            "fit_start_time": train_inter[0],
            "fit_end_time": train_inter[1],
            "instruments": instruments,
        }

        # 模型和数据的配置参数
        task = {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
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
                        "train": train_inter,
                        "valid": valid_inter,
                        "test": test_inter,
                    },
                },
            },
        }

        # model initialization
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])

        # 回测配置信息
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
                    "topk": 20,
                    "n_drop": 2,
                },
            },
            "backtest": {
                "start_time": test_inter[0],
                "end_time": test_inter[1],
                "account": 50000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0003,
                    "close_cost": 0.0013,
                    "min_cost": 5,
                },
            },
        }

        # 训练
        with R.start(experiment_name=self.experiment_name):
            R.log_params(**flatten_dict(task))
            model.fit(dataset)
            # 保存模型
            R.save_objects(**{'params.pkl': model})

            recorder = R.get_recorder()
            # 预测信号
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # 信号分析
            sar = SigAnaRecord(recorder, ana_long_short=True)
            sar.generate()

            # 回测
            par = PortAnaRecord(recorder, config=port_analysis_config, risk_analysis_freq='day')
            par.generate()

if __name__ == '__main__':
    fire.Fire(LightGBMAlpha158)
