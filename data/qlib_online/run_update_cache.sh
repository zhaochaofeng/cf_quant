#!/bin/bash
source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
cf_quant_path="${CF_QUANT_PATH}"
echo 'cur_path: '${cur_path}

if [ $# -eq 0 ];then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ];then
    dt=$1
  else
    echo "输入参数错误"
    exit 1
fi
echo "dt: "${dt}


${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo "非交易日"
  exit 0
fi


${python_path} ${cur_path}/update_cache.py update_all --max_workers 10 --early_stop_failures 1


if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
