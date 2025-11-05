#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
echo 'cur_path: '${cur_path}

if [ $# -eq 0 ];then
  dt=`date +%Y-%m-%d`
elif [ $# -eq 1 ];then
  dt=$1
else
  echo "输入参数错误！！！"
  exit 1
fi
echo "dt: "${dt}

${python_path} ${cur_path}/check.py --now_date ${dt}

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
