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
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel as TransformerModelTS


class LGBModel2(LGBModel):

    def _prepare_data_finetune(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        """
        Prepare data_new for finetune
        """
        ds_l = []
        assert "valid" in dataset.segments
        assert "test" in dataset.segments
        # for key in ["train", ["valid", "test"]]:
        for key in [["valid", "test"]]:
            # if key in dataset.segments:
            df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            if isinstance(df, list):  # 合并 valid 和 test
                df = pd.concat(df, axis=0, join="outer")
            if df.empty:
                raise ValueError("Empty data_new from dataset, please check your dataset config.")
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
        ds_l = self._prepare_data_finetune(dataset, reweighter)  # pylint: disable=W0632
        ds, names = list(zip(*ds_l))
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(
            self.params,
            train_set=ds[0],
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=ds,
            valid_names=["train"],
            callbacks=[verbose_eval_callback],
        )


class TransformerModel2(TransformerModel):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.device = get_device()
        self.model.to(self.device)


class TransformerModelTS2(TransformerModelTS):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.device = get_device()
        self.model.to(self.device)
    
    def train_epoch(self, data_loader):
        self.model.train()

        for data in data_loader:
            # Ensure data is converted to float32 before moving to device
            feature = data[:, :, 0:-1].float().to(self.device)
            label = data[:, -1, -1].float().to(self.device)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            
    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:
            # mps 仅支持float32，需要将float64转换为float32
            feature = data[:, :, 0:-1].float().to(self.device)
            label = data[:, -1, -1].float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)


def get_device(GPU=0, return_str=False):
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


def init_model(model, **kwargs):
    return model(**kwargs)


def prepare_model_config(
        class_name: str = 'LGBModel',
        module_path: str = 'qlib.contrib.model',
        **kwargs):
    model = {
        'class': class_name,
        'module_path': module_path,
        'kwargs': kwargs
    }
    return model


if __name__ == '__main__':
    from pprint import pprint
    pprint(prepare_model_config())
