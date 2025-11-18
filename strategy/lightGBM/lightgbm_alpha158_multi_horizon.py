'''
    功能：lightGBM_Alpha158 多周期策略的模型训练
'''
import copy
import time
import fire
import pandas as pd
from typing import Union
import qlib
from qlib.data import D
from qlib.workflow import R
from qlib.data.filter import NameDFilter
from qlib.utils import flatten_dict
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.loader import DataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.task.gen import MultiHorizonGenBase
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils.data import zscore
from datetime import datetime, timedelta
from utils.utils import sql_engine
from utils.utils import get_n_pretrade_day
from utils.utils import send_email
import traceback

class Alpha158MultiHorizonGen(MultiHorizonGenBase):
    def set_horizon(self, t: dict, hr: int):
        t.setdefault("extra", {})["horizon"] = hr

class CustomDataLoader(DataLoader):
    def __init__(self, data):
        self.data = data
    def load(self, instruments, start_time=None, end_time=None):
        return self.data

class LightGBMAlpha158:
    def __init__(self,
        market='all',
        benchmark="SH000300",
        provider_uri='~/.qlib/qlib_data/custom_data_hfq',
        uri=None,
        experiment_name='lightGBM_Alpha158',
        start_wid=1,        # test_end 向前移动的天数。至少前移1天，保证回测时不出错
        test_wid=100,       # 测试集时间宽度
        valid_wid=100,      # 验证集时间宽度
        train_wid=500,      # 训练集时间宽度
        horizon=None        # 预测时间跨度
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

        if uri is None:
            uri = './mlruns'
        self.uri = uri
        self.market = market
        self.benchmark = benchmark
        self.experiment_name = experiment_name
        self.start_wid = start_wid
        self.test_wid = test_wid
        self.valid_wid = valid_wid
        self.train_wid = train_wid
        self.horizon = horizon
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
        sql = """select ts_code, name, list_date from cf_quant.stock_info_ts where day>='{}' and day<='{}'""".format(dt1, dt2)
        stocks_info = pd.read_sql(sql, engine)
        # 过滤ST、退市和次新股
        stocks_filter = stocks_info[
                (stocks_info['name'].str.contains('ST')) |      # ST股票
                (stocks_info['name'].str.contains('退')) |      # 退市股票
                (stocks_info['list_date'] > datetime.strptime(target_date, '%Y-%m-%d').date())        # 次新股
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

    def dataframe_to_dataseth(self, df, segments):
        cus_dataloader = CustomDataLoader(df)
        data_handler = DataHandlerLP(
            data_loader=cus_dataloader
        )
        return DatasetH(handler=data_handler, segments=segments)

    def get_label(self, start_time, end_time, hr, instruments):
        col_name = f'LABEL{hr}'
        fields = [f"Ref($close, -({hr}+1)) / Ref($close, -1) - 1"]
        df = D.features(instruments=instruments, fields=fields, start_time=start_time, end_time=end_time)
        df.columns = pd.MultiIndex.from_tuples([('label', col_name)])

        df_learn = copy.deepcopy(df)   # 用于训练(learn)
        df_learn.dropna(inplace=True, axis=0)
        df_learn = df_learn.groupby('datetime', group_keys=False).apply(zscore)
        df_learn = df_learn.swaplevel()   # 索引转化为<datetime, instrument>
        df_learn.sort_index(inplace=True)

        df = df.swaplevel()     # 用于推断（infer）
        df.sort_index(inplace=True)
        return df_learn, df

    def main(self):
        train_inter, valid_inter, test_inter = self.date_interval()
        instruments = self.choose_stocks(train_inter[0], test_inter[1])

        data_handler_config = {
            "start_time": train_inter[0],
            "end_time": test_inter[1],
            "fit_start_time": train_inter[0],
            "fit_end_time": train_inter[1],
            "instruments": instruments,
            # "drop_raw": True,
            "learn_processors": []
        }

        # 基础任务模板（标签将由MultiHorizon生成器按周期覆盖）
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
                    "num_threads": 10,
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

        dataset = init_instance_by_config(task["dataset"])
        # dataset_config = copy.deepcopy(dataset)  # 复制实例，此时没有处理实际数据
        features_learn = pd.concat(dataset.prepare(segments=['train', 'valid', 'test'], col_set=['feature'], data_key='learn'), axis=0)
        features_infer = pd.concat(dataset.prepare(segments=['train', 'valid', 'test'], col_set=['feature'], data_key='infer'), axis=0)

        mh_gen = Alpha158MultiHorizonGen(horizon=self.horizon, label_leak_n=2)
        tasks = mh_gen.generate(task)

        assert len(tasks) == len(self.horizon)
        # 逐周期训练与回测
        for i, hr in enumerate(self.horizon):
            print('{}\nHorizon: {}'.format('-' * 100, hr))
            model = init_instance_by_config(task["model"])

            # label_col = "LABEL{}".format(hr)
            # label = labels.loc[:, (slice(None), label_col)]
            label_learn, label_infer = self.get_label(task['dataset']['kwargs']['handler']['kwargs']['start_time'],
                                   task['dataset']['kwargs']['handler']['kwargs']['end_time'],
                                   hr, instruments)

            data_learn = pd.concat([features_learn, label_learn], axis=1, join='inner')
            data_infer = pd.concat([features_infer, label_infer], axis=1, join='inner')

            segs = tasks[i]["dataset"]["kwargs"]["segments"]
            print("segments: {}".format(segs))
            dataset_learn = self.dataframe_to_dataseth(data_learn, segs)
            dataset_infer = self.dataframe_to_dataseth(data_infer, segs)

            print('dataset_learn: {}'.format('-' * 100))
            print("features_learn shape: {}".format(features_learn.shape))
            print("label_learn shape: {}".format(label_learn.shape))
            print(dataset_learn.prepare(col_set=['feature', 'label'], segments='train'))
            print('dataset_infer: {}'.format('-' * 100))
            print("features_infer shape: {}".format(features_infer.shape))
            print("label_infer shape: {}".format(label_infer.shape))
            print(dataset_infer.prepare(col_set=['feature', 'label'], segments='train'))

            # 使用生成器截断后的测试区间，保证与标签一致
            bt_start, bt_end = segs["test"][0], segs["test"][1]

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
                        "dataset": dataset_infer,
                        "topk": 20,
                        "n_drop": 2,
                        "hold_thresh": hr
                    },
                },
                "backtest": {
                    "start_time": bt_start,
                    "end_time": bt_end,
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

            exp_name = f"{self.experiment_name}_h{hr}"
            with R.start(experiment_name=exp_name, uri=self.uri):
                R.log_params(**flatten_dict(task))
                model.fit(dataset_learn)
                R.save_objects(**{'params.pkl': model})
                R.save_objects(**{'task': task})
                R.save_objects(**{'dataset': dataset})

                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset_infer, recorder)
                sr.generate()

                sar = SigAnaRecord(recorder, ana_long_short=True)
                sar.generate()

                par = PortAnaRecord(recorder, config=port_analysis_config, risk_analysis_freq='day')
                par.generate()

        del label_learn, label_infer
        del data_learn, data_infer
        # 先删除持有 dataset_infer 引用的对象
        del par  # par 持有 port_analysis_config，port_analysis_config 持有 dataset_infer
        del sr   # sr 直接持有 dataset_infer
        del port_analysis_config  # 删除包含 dataset_infer 引用的配置字典
        # 再删除 dataset_infer 变量名（此时如果没有其他引用，对象才会被释放）
        del dataset_learn, dataset_infer
        del model, sar


if __name__ == '__main__':
    try:
        t = time.time()
        fire.Fire(LightGBMAlpha158)
        print('耗时：{}'.format(round(time.time()-t, 4)))
    except Exception as e:
        print('error: {}'.format(e))
        error_info = traceback.format_exc()
        send_email('Strategy: lightgbm_alpha158', error_info)

    '''
        python lightgbm_alpha158_multi_horizon.py main --start_wid 2 --train_wid 100 --horizon 1,2,3
    '''
