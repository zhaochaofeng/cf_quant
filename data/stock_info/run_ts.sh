#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
echo 'cur_path: '${cur_path}

dt=`date +%Y-%m-%d`
echo "dt: "${dt}

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo '非交易日，退出！'
  exit 0
fi

${python_path} ${cur_path}/stock_info_ts2.py ${dt}

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
