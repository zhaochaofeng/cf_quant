'''
    数据提供框架
'''

from typing import Callable, Union

import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from qlib.data.dataset import DatasetH, DataHandler, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader
from qlib.utils import init_instance_by_config
from qlib.workflow.task.gen import MultiHorizonGenBase

from utils import multiprocessing_wrapper
from utils import (
    standardize, winsorize
)


class MultiHorizonGen(MultiHorizonGenBase):
    """
        多周期预测
    """

    def set_horizon(self, t: dict, hr: int):
        t.setdefault("extra", {})["horizon"] = hr


class CustomDataLoader(DataLoader):
    """
        自定义DataLoder
    """

    def __init__(self, data):
        self.data = data

    def load(self, instruments, start_time=None, end_time=None):
        return self.data


def dataframe_to_dataset(df, segments: dict) -> DatasetH:
    """
        将DataFrame 转化为 DatasetH
    Parameters
    ----------
    df: DataFrame
    segments: train/valid/test 划分区间

    Returns： DatasetH
    -------

    """
    cus_dataloader = CustomDataLoader(df)
    data_handler = DataHandlerLP(
        data_loader=cus_dataloader
    )
    return DatasetH(handler=data_handler, segments=segments)


class ExpAlpha158(Alpha158):
    def __init__(self,
                 expand_feas: tuple[list, list] = None,
                 ref: int = -2,
                 **kwargs
                 ):
        """
        Args:
            expand_feas: 扩展特征。(fields, name)
            is_win: 是否取极值
            is_std: 是否标准化
        """
        self.expand_feas = expand_feas
        self.ref = ref
        super().__init__(**kwargs)

    def get_label_config(self):
        return [f"Ref($close, {self.ref})/Ref($close, -1) - 1"], ["LABEL0"]

    def get_feature_config(self):
        # 获取原始Alpha158特征配置
        fields, names = super().get_feature_config()
        # 添加自定义特征
        if self.expand_feas:
            fields.extend(self.expand_feas[0])
            names.extend(self.expand_feas[1])
        # 删除 VWAP0 列
        # idx = names.index("VWAP0")
        # fields.pop(idx)
        # names.pop(idx)
        return fields, names


def construct_data_for_griffinnet():
    import pickle
    import os
    import qlib
    from qlib.data.dataset import TSDatasetH
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
    # segments = {
    #     "train": ("2008-01-01", "2020-03-31"),
    #     "valid": ("2020-04-15", "2020-06-30"),  # 集合之间相距15天，防止数据泄漏
    #     "test": ("2020-07-15", "2022-12-31"),
    # }

    segments = {
        "train": ("2011-01-01", "2023-05-31"),
        "valid": ("2023-06-15", "2023-12-31"),
        "test": ("2024-01-15", "2025-10-31"),
    }

    # segments = {
    #     "train": ("2020-01-01", "2020-03-31"),
    #     "valid": ("2020-04-15", "2020-06-30"),
    #     "test": ("2020-07-15", "2020-08-31"),
    # }
    handler_config = {
        "start_time": segments['train'][0],
        "end_time": segments['test'][1],
        "fit_start_time": segments['train'][0],
        "fit_end_time": segments['train'][1],
        "instruments": "csi300",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
            # {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "Fillna", "kwargs": {}},
            {"class": "ProcessInf", "kwargs": {}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"}
        ]
    }

    exp_fields = []
    for s in ['SH000300', 'SH000905', 'SH000906']:
        f = "Mask($close / Ref($close, 1)-1, '{}')".format(s)
        exp_fields.append(f)
        for d in [5, 10, 20, 30, 60]:
            m1 = "Mask(Mean($close / Ref($close, 1)-1, {}), '{}')".format(d, s)
            s1 = "Mask(Std($close / Ref($close, 1)-1, {}), '{}')".format(d, s)
            m2 = f"Mask(Mean($amount, {d}) / ($amount), '{s}')"
            s2 = f"Mask(Std($amount, {d}) / ($amount), '{s}')"
            exp_fields.extend([m1, s1, m2, s2])
    exp_name = ['f_{}'.format(i + 1) for i in range(len(exp_fields))]
    handler = ExpAlpha158(
        expand_feas=(exp_fields, exp_name),
        **handler_config
    )
    tsdh = TSDatasetH(
        handler=handler,
        segments=segments,
        step_len=8
    )

    train = tsdh.prepare(segments='train', data_key='learn')
    valid = tsdh.prepare(segments='valid', data_key='learn')
    test = tsdh.prepare(segments='test', data_key='infer')
    train.config(**{'fillna_type': 'ffill+bfill'})
    valid.config(**{'fillna_type': 'ffill+bfill'})
    test.config(**{'fillna_type': 'ffill+bfill'})

    path_base = 'griffinnet/custom_data_new'
    os.makedirs(path_base, exist_ok=True)
    with open(os.path.join(path_base, 'csi300_dl_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(path_base, 'csi300_dl_valid.pkl'), 'wb') as f:
        pickle.dump(valid, f)
    with open(os.path.join(path_base, 'csi300_dl_test.pkl'), 'wb') as f:
        pickle.dump(test, f)


def prepare_data(segments: dict,
                 handler_model: DataHandler = Alpha158,
                 instruments: str = 'csi300',
                 learn_processors: list = None,
                 infer_processors: list = None,
                 data_type: str = 'ds',
                 step_len: int = None,
                 **kwargs
                 ) -> (Union[DatasetH, TSDatasetH], dict):
    """
    Args:
        segments: 训练集、验证集、测试集的划分
        handler_model: 数据处理模型
        instruments: 使用的股票池
        learn_processors: 训练集数据处理
        infer_processors: 测试集数据处理
        data_type: 数据类型。ds: DatasetH; ts: TSDatasetH
        step_len: 时间步长。ts==TSDatasetH 时有效
    """
    if data_type == 'ts' and step_len is None:
        raise ValueError('step_len is required when data_type is ts')
    if learn_processors is None:
        learn_processors = []
    if infer_processors is None:
        infer_processors = []

    data_handler_config = {
        'start_time': segments['train'][0],
        'end_time': segments['test'][1],
        'fit_start_time': segments['train'][0],
        'fit_end_time': segments['train'][1],
        'instruments': instruments,
        'learn_processors': learn_processors,
        'infer_processors': infer_processors,
    }
    data_handler_config.update(kwargs)

    handler = handler_model(**data_handler_config)
    kwargs = {
        'handler': handler,
        'segments': segments
    }
    if data_type == 'ts':
        kwargs.update({'step_len': step_len})

    config = {
        'class': 'DatasetH' if data_type == 'ds' else 'TSDatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': kwargs
    }
    dataset = init_instance_by_config(config)
    return dataset, config


def prepare_data_config(
        segments: dict,
        class_name: str = 'Alpha158',
        module_path: str = 'qlib.contrib.data.handler',
        instruments: str = 'csi300',
        learn_processors: list = None,
        infer_processors: list = None,
        data_type: str = 'ds',
        step_len: int = None,
        **kwargs
    ) -> dict:
    """
    Args:
        segments: 训练集、验证集、测试集的划分
        class_name: 数据类
        module_path: 数据类路径
        instruments: 使用的股票池
        learn_processors: 训练集数据处理
        infer_processors: 测试集数据处理
        data_type: 数据类型。ds: DatasetH; ts: TSDatasetH
        step_len: 时间步长。ts==TSDatasetH 时有效
        **kwargs: 用于传递给handler的参数

    Example:
        if __name__ == '__main__':
            import qlib
            qlib.init()
            from qlib.utils import init_instance_by_config
            from strategy.dataset import prepare_data_config
            from pprint import pprint

            segments = {
                "train": ("2018-01-01", "2018-03-31"),
                "valid": ("2018-04-01", "2018-05-31"),
                "test": ("2018-06-01", "2018-09-10"),
            }
            dataset = prepare_data_config(segments=segments, class_name='ExpAlpha158', module_path='strategy.dataset')
            pprint(dataset)
            dataset = init_instance_by_config(dataset)
            print(dataset.prepare(segments='train'))
    """
    if data_type == 'ts' and step_len is None:
        raise ValueError('step_len is required when data_type is ts')
    if learn_processors is None:
        learn_processors = []
    if infer_processors is None:
        infer_processors = []

    data_handler_config = {
        'start_time': segments['train'][0],
        'end_time': segments['test'][1],
        'fit_start_time': segments['train'][0],
        'fit_end_time': segments['train'][1],
        'instruments': instruments,
        'learn_processors': learn_processors,
        'infer_processors': infer_processors,
    }
    data_handler_config.update(kwargs)

    dataset_kwargs = {
        'handler': {
            'class': class_name,
            'module_path': module_path,
            'kwargs': data_handler_config
        },
        'segments': segments
    }
    if data_type == 'ts':
        dataset_kwargs.update({'step_len': step_len})

    dataset = {
        'class': 'DatasetH' if data_type == 'ds' else 'TSDatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': dataset_kwargs
    }
    return dataset


def calculate_and_merge_factors(
        factor_funcs: list[Callable],
        fields: list[str],
        instruments: Union[list, dict],
        start_time: str,
        end_time: str,
        n: int = 1
) -> Union[pd.DataFrame, None]:
    """
        计算因子值并按列合并
    Parameters
    ----------
    factor_funcs: 因子函数列表
    fields: 原始字段列表。如[$close]
    instruments: 股票列表或配置
    start_time: 起始日期
    end_time: 终止日期
    is_save: 是否保存为h5文件
    is_return: 是否返回合并和的因子DataFrame
    n: 并行数

    Returns
    -------
        DataFrame Or None

    Example
    ------
    import qlib
    from data.factor import MOM_10D, VW_MOM_5D

    factor_funcs = [MOM_10D, VW_MOM_5D]

    provider_uri = '~/.qlib/qlib_data/custom_data_hfq'
    qlib.init(provider_uri=provider_uri)

    fields = ['$close', '$volume']
    res = calculate_and_merge_factors(
        factor_funcs=factor_funcs,
        fields=fields,
        instruments=D.instruments(market='csi300'),
        start_time="2025-11-01",
        end_time="2025-12-31",
        is_save=True,
        is_return=True,
        n=2
    )
    print(res)

    """
    df = D.features(instruments=instruments,
                    fields=fields,
                    start_time=start_time,
                    end_time=end_time)

    res_list = multiprocessing_wrapper([(factor, (df,)) for factor in factor_funcs], n=n)
    res_df = pd.concat(res_list, axis=1, join='outer')
    res_df.sort_index(inplace=True)
    return res_df


if __name__ == '__main__':
    segments = {
        "train": ("2018-01-01", "2018-03-31"),
        "valid": ("2018-04-01", "2018-05-31"),
        "test": ("2018-06-01", "2018-09-10"),
    }
    from pprint import pprint
    pprint(prepare_data_config(segments))


