#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
uri='/data/cf_quant/mlruns'

if [ $# -eq 0 ];then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ];then
    dt=$1
  else
    echo "参数错误"
    exit 1
fi

echo ${cur_path}
echo "dt: "${dt}

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi

${python_path} ${cur_path}/predict.py main \
--start_date ${dt} \
--end_date ${dt} \
--uri ${uri} \
--horizon 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

if [ $? -eq 0 ]; then
  echo "执行成功！"
else
  echo "执行失败！"
  exit 1
fi
