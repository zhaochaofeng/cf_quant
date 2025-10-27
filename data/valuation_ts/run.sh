#!/bin/bash

source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
echo ${cur_path}

if [ $# -eq 0 ]; then
    dt1=`date +%Y-%m-%d`
    dt2=${dt1}
  elif [ $# -eq 1 ]; then
    dt1=$1
    dt2=${dt1}
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  else
    echo "参数错误"
    exit 1
fi

${python_path} ${cur_path}/valuation_ts.py --start_date ${dt1} --end_date ${dt2}

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
