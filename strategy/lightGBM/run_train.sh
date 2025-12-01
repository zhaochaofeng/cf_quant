#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
provider_uri="/root/.qlib/qlib_data/custom_data_hfq"
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
dt=`date +%Y-%m-%d`

echo ${cur_path}
echo "dt: "${dt}

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi

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

${python_path} ${cur_path}/lightgbm_alpha.py \
  --provider_uri "${provider_uri}" \
  --instruments "csi300" \
  --is_online False \
  --horizon "[1,2,3,4,5]" \
  --start_wid "${start_wid}" \
  --test_wid 200 \
  --valid_wid 100 \
  --train_wid 800 \
  main


if [ $? -eq 0 ]; then
  echo "执行成功！"
else
  err_msg="执行失败！"
  echo ${err_msg}
  ${python_path} ${cf_quant_path}/utils/send_email.py "Strategy: lightgbm_alpha" "${err_msg}"
  exit 1
fi

