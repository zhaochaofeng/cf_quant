#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
cf_quant_path="/root/cf_quant"
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
    echo "参数数量最多为2 !"
    exit 1
fi

echo "时间区间为：["$dt1" - "$dt2"]"

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt2}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi


#curl -X 'POST' \
#  'http://localhost:8000/download_to_redis' \
#  -H 'accept: application/json' \
#  -H 'Content-Type: application/json' \
#  -d "{
#    \"stock_codes\": [],
#    \"start_date\": \"${dt1}\",
#    \"end_date\": \"${dt2}\"
#  }"
#

cmd="curl -X 'POST' 'http://localhost:8000/download_to_redis' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{\"stock_codes\":[],\"start_date\":\"${dt1}\",\"end_date\":\"${dt2}\"}'"

echo "执行命令: $cmd"
eval $cmd

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  ${python_path} ${cf_quant_path}/utils/send_email.py 'Stragegy: lightgbm_alpha158' 'run_predict fail'
  exit 1
fi
