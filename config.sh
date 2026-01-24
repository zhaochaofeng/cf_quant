#!/bin/bash
# 全局配置文件 - 统一管理项目环境变量

# Python解释器路径
export PYTHON_PATH="/home/data/anaconda3/envs/python3/bin/python"

# 项目路径
export CF_QUANT_PATH="/home/data/cf_quant"
export QLIB_PATH="/home/data/qlib"

# 数据路径
export QLIB_DATA_PATH="/chaofeng/.qlib/qlib_data"
export PROVIDER_URI="${QLIB_DATA_PATH}/custom_data_hfq"

# MLflow路径
export MLFLOW_URI="/home/data/cf_quant/mlruns"

