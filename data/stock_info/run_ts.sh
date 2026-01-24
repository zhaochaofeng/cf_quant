#!/bin/bash
source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
cf_quant_path="${CF_QUANT_PATH}"
echo 'cur_path: '${cur_path}

dt=`date +%Y-%m-%d`
echo "dt: "${dt}

${python_path} ${cur_path}/stock_info_ts.py --now_date ${dt} main

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
