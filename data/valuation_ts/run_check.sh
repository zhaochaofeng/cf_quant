#!/bin/bash

source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
echo "cur_path: "${cur_path}

if [ $# -eq 0 ]; then
    dt2=`date +%Y-%m-%d`
    dt1=`date -d "-5 days $dt2" +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt2=$1
    dt1=`date -d "-5 days $dt2" +%Y-%m-%d`
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  else
    echo "参数错误"
    exit 1
fi

echo "[${dt1} - ${dt2}]"

${python_path} ${cur_path}/check.py --start_date ${dt1} --end_date ${dt2}

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
