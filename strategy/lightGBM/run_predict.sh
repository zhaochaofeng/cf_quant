#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
cf_quant_path="/root/cf_quant"
uri='/data/cf_quant/mlruns'

if [ $# -eq 0 ];then
    dt1=`date +%Y-%m-%d`
    dt2=${dt1}
  elif [ $# -eq 1 ];then
    dt1=$1
    dt2=${dt1}
  elif [ $# -eq 2 ];then
    dt1=$1
    dt2=$2
  else
    echo "参数错误"
    exit 1
fi

echo "cur_path: ${cur_path}"
echo "[${dt1} - ${dt2}]"

${python_path} ${cf_quant_path}/utils/is_trade_day.py "${dt2}"
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi


check_success(){
  # 执行结果检查函数
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    err_msg="$1 执行失败！！！"
    echo "${err_msg}"
    exit 1
  fi
}


#${python_path} ${cur_path}/predict.py main \
#  --start_date "${dt1}" \
#  --end_date "${dt2}" \
#  --uri "${uri}" \
#  --instruments "csi300" \
#  --exp_name "lightgbm_alpha_csi300" \
#  --horizon "[1,2,3,4,5]"
#check_success "csi300"


${python_path} ${cur_path}/predict.py main \
  --start_date "${dt1}" \
  --end_date "${dt2}" \
  --uri "${uri}" \
  --instruments "csi500" \
  --exp_name "lightgbm_alpha_csi500" \
  --horizon "[1,2,3,4,5]"
check_success "csi500"


#${python_path} ${cur_path}/predict.py main \
#  --start_date "${dt1}" \
#  --end_date "${dt2}" \
#  --uri "${uri}" \
#  --instruments "csia500" \
#  --exp_name "lightgbm_alpha_csia500" \
#  --horizon "[1,2,3,4,5]"
#check_success "csia500"

