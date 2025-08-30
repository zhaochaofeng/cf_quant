#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
echo ${cur_path}

if [ $# -eq 0 ];then
    start_wid=1
  elif
    [ $# -eq 1 ];then
    start_wid=$1
  else
    echo "参数错误"
    exit 1
fi

echo 'start_wid: '${start_wid}

${python_path} ${cur_path}/lightgbm_alpha158.py main --start_wid ${start_wid}

if [ $? -eq 0 ]; then
  echo "执行成功！"
else
  echo "执行失败！"
  exit 1
fi
