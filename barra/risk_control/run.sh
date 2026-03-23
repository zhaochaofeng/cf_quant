#!/bin/bash

source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
project_path="${CF_QUANT_PATH}"
echo "cur_path: ${cur_path}"
echo "python_path: ${python_path}"
echo "project_path: ${project_path}"


if [ $# -eq 0 ]; then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt=$1
  else
    echo "参数错误"
    exit 1
fi


function is_trade_day(){
  dt=$1
  ${python_path} ${project_path}/utils/is_trade_day.py $dt
  if [ $? -eq 5 ];then
    echo "非交易日: ${dt}"
    exit 0
  fi
}

function check() {
  # 判断是否执行成功
  if [ $? -eq 0 ]; then
    echo "${name} 执行成功！！！"
   else
    echo "${name}执行失败！！！"
    exit 1
  fi
}

is_trade_day ${dt}

${python_path} ${cur_path}/run_monthly.py --end-date ${dt}
echo "run_monthly: "
check
${python_path} ${cur_path}/run_daily.py
echo "run_daily"
check





