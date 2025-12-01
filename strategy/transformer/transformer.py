
import fire
import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from strategy.dataset import prepare_data
# from strategy.model import TransformerModelTS2
from strategy.model import TransformerModel2



def main():
    qlib.init()

    benchmark = 'SH000300'
    instruments = 'csi300'

    segments = {
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
    }
    # step_len = 20
    learn_processors = [
        {'class': 'DropnaLabel'},
        {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}
    ]
    infer_processors = [
        {'class': 'FilterCol', 'kwargs': {'fields_group': 'feature',
                                          'col_list': ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5",
                                                       "CORR10", "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60",
                                                       "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
                                          }
         },
        {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
        {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}
    ]

    # dataset = prepare_data(segments, step_len=step_len, data_type='ts', instruments=instruments,
    #                        learn_processors=learn_processors, infer_processors=infer_processors)
    dataset = prepare_data(segments, data_type='ds', instruments=instruments,
                           learn_processors=learn_processors, infer_processors=infer_processors)
    # model = TransformerModelTS2(d_feat=20, seed=0, n_jobs=5)
    model = TransformerModel2(d_feat=20, seed=0, dropout=0.15)
    param_num = sum(p.numel() for p in model.model.parameters())
    print('\n{}\n{}'.format('-' * 100, param_num))

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
            },
        },
        "backtest": {
            "start_time": segments['test'][0],
            "end_time": segments['test'][1],
            "account": 100000000,
            "benchmark": benchmark,
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

    with R.start(experiment_name='test'):
        model.fit(dataset)
        R.save_objects(**{'params.pkl': model})
        R.save_objects(**{'dataset': dataset})

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, config=port_analysis_config, risk_analysis_freq='day')
        par.generate()


if __name__ == '__main__':
    fire.Fire(main)



