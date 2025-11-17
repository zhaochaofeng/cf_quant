#!/bin/bash
source ~/.bashrc
cur_path=$(pwd)
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
uri='/data/cf_quant/mlruns'

dt=$(date +%Y-%m-%d)

if [ $# -eq 0 ]; then
    start_wid=1
  elif [ $# -eq 1 ]; then
    start_wid=$1
  elif [ $# -eq 2 ]; then
    start_wid=$1
    dt=$2
  else
    echo "参数错误"
    exit 1
fi

echo "${cur_path}"
echo "start_wid: ${start_wid}"
echo "dt: ${dt}"


${python_path} ${cf_quant_path}/utils/is_trade_day.py "${dt}"
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi



${python_path} ${cur_path}/lightgbm_alpha158_multi_horizon.py main \
--start_wid "${start_wid}" \
--train_wid 500 \
--uri ${uri} \
--horizon 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

exit_code=$?

if [ ${exit_code} -eq 0 ]; then
    echo "执行成功！"
  elif [ ${exit_code} -eq 137 ];then
    msg="程序被kill(可能是OOM), exit_code: ${exit_code}"
    echo "${msg}"
    ${python_path} ${cf_quant_path}/utils/send_email.py "Strategy: ligthGBM: train_multi" "${msg}"
  else
    msg="执行失败！"
    echo ${msg}
    ${python_path} ${cf_quant_path}/utils/send_email.py "Strategy: ligthGBM: train_multi" "${msg}"
    exit 1
fi
