'''
    模型框架
'''
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.contrib.model import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
import torch
from qlib.contrib.model.pytorch_transformer import TransformerModel


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

class TransformerModel2(TransformerModel):
    def __init__(self, dropout=0.15, GPU=0, d_feat=359, seed=0):
        super().__init__(dropout=dropout, GPU=GPU, d_feat=d_feat, seed=seed)
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


