#!/bin/bash

source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
echo ${cur_path}

if [ $# -eq 0 ]; then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt=$1
  else
    echo "参数错误"
    exit 1
fi

function check() {
  name=$1
  # 判断是否执行成功
  if [ $? -eq 0 ]; then
    echo "${name} 执行成功！！！"
    exit 0
   else
    echo "${name}执行失败！！！"
    exit 1
  fi
}

${python_path} ${cur_path}/run_monthly.py --end-date ${dt}
check "run_monthly"
${python_path} ${cur_path}/run_daily.py
check "run_daily"





