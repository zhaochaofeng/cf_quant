#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
provider_uri="/root/.qlib/qlib_data/custom_data_hfq"
uri="/data/cf_quant/mlruns"
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

echo "start_wid: ${start_wid}"


check_success(){
  # 执行结果检查函数
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    err_msg="$1 执行失败！！！"
    echo "${err_msg}"
    ${python_path} ${cf_quant_path}/utils/send_email.py "Data: index" "${err_msg}"
    exit 1
  fi
}


${python_path} ${cur_path}/lightgbm_alpha.py \
  --provider_uri "${provider_uri}" \
  --uri "${uri}" \
  --instruments "csi300" \
  --exp_name "lightgbm_alpha_csi300" \
  --is_online False \
  --horizon "[1,2,3,4,5]" \
  --start_wid "${start_wid}" \
  --test_wid 252 \
  --valid_wid 100 \
  --train_wid 800 \
  main
check_success "csi300"


${python_path} ${cur_path}/lightgbm_alpha.py \
  --provider_uri "${provider_uri}" \
  --uri "${uri}" \
  --instruments "csi500" \
  --exp_name "lightgbm_alpha_csi500" \
  --is_online False \
  --horizon "[1,2,3,4,5]" \
  --start_wid "${start_wid}" \
  --test_wid 252 \
  --valid_wid 100 \
  --train_wid 800 \
  main
check_success "csi500"


${python_path} ${cur_path}/lightgbm_alpha.py \
  --provider_uri "${provider_uri}" \
  --uri "${uri}" \
  --instruments "csia500" \
  --exp_name "lightgbm_alpha_csia500" \
  --is_online False \
  --horizon "[1,2,3,4,5]" \
  --start_wid "${start_wid}" \
  --test_wid 252 \
  --valid_wid 100 \
  --train_wid 800 \
  main
check_success "csia500"
