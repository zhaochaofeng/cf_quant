#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
echo 'cur_path: '${cur_path}

if [ $# -eq 0 ]; then
    dt2=`date +%Y-%m-%d`
    dt1=`date -d "5 days ago" +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt2=$1
    dt1=`date -d "5 days ago" +%Y-%m-%d`
  elif [ $# -eq 2 ]; then
    dt2=$1
    dt1=$2
  else
    echo "参数错误"
    exit 1
fi

echo "[${dt1} - ${dt2}]"

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo "非交易日"
  exit 0
fi


${python_path} ${cur_path}/check_cache.py --start_date "${dt1}" --end_date "${dt2}"


if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
