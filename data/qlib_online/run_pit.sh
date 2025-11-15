#!/bin/bash

# 功能：权量更新 财务数据（PIT）
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
cf_quant_path="/root/cf_quant"
data_path="/root/.qlib/qlib_data"
provider_uri="${data_path}/custom_data_hfq"

echo "cur_path: ${cur_path}"

if [ $# -eq 0 ]; then
    dt1='2015-01-05'
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt1=$1
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  elif [ $# -eq 3 ]; then
    dt1=$1
    dt2=$2
    provider_uri=$3
  else
    echo "参数错误！！！"
    exit 1
fi

echo "时间区间为：[$dt1 - $dt2]"
echo "provider_uri: ${provider_uri}"

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt2}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi


check_success(){
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    echo "$1 执行失败！！！"
    exit 1
  fi
}

get_data_from_mysql(){
  echo "get_data_from_mysql ..."

  sql=$(cat <<-EOF
    SELECT
      t.f_ann_date AS date,
      t.end_date AS period,
      lower(t.qlib_code) AS symbol,
      jt.field,  -- 正确显示字段名：n_income 或 n_income_attr_p
      -- 根据字段名提取对应的值（自动适配数值类型）
      JSON_EXTRACT(
        JSON_OBJECT(
          'n_income', t.n_income,
          'n_income_attr_p', t.n_income_attr_p
          -- 新增字段只需在这里添加：'新字段名', t.新字段
        ),
        CONCAT('$.', jt.field)  -- 构建路径：$.n_income 或 $.n_income_attr_p
      ) AS value
    FROM
      cf_quant.income_ts t,
      -- 解析 JSON 键名数组，得到 field 字段
      JSON_TABLE(
        -- 生成键名数组：["n_income", "n_income_attr_p"]
        JSON_KEYS(JSON_OBJECT(
          'n_income', t.n_income,
          'n_income_attr_p', t.n_income_attr_p
        )),
        '\$[*]' COLUMNS (  -- 遍历数组中的每个键名。需要加斜杠转译符
          field VARCHAR(50) PATH '$'  -- 提取键名作为 field
        )
      ) AS jt
    WHERE
      t.f_ann_date >= '${dt1}'
      AND t.f_ann_date <= '${dt2}';
EOF
)
  echo "${sql}"
  mysql -uchaofeng -pZhao_123 -e "${sql}" > "${provider_uri}/pit_${dt1}_${dt2}.csv"
  check_success "从mysql中导出数据"
}


split_stock(){
  echo "split_stock ..."
  ${python_path} "${cur_path}"/split_stock.py \
  --path_in "${provider_uri}/pit_${dt1}_${dt2}.csv" \
  --path_out "${provider_uri}" \
  --start_date "${dt1}" \
  --end_date "${dt2}"
  check_success "切分股票数据"
}


format(){
  echo "format ..."
  ${python_path} ${qlib_path}/scripts/data_collector/pit/collector.py normalize_data \
  --interval quarterly \
  --source_dir "${provider_uri}/pit_${dt1}_${dt2}" \
  --normalize_dir "${provider_uri}/pit_normalized_${dt1}_${dt2}"
  check_success "格式化"
}

trans_to_qlib(){
  echo "trans_to_qlib..."
  ${python_path} ${qlib_path}/scripts/dump_pit.py dump \
  --csv_path "${provider_uri}/pit_normalized_${dt1}_${dt2}" \
  --qlib_dir "${provider_uri}" \
  --interval quarterly
  check_success "转化为qlib格式"
}


function_set(){
  get_data_from_mysql
  split_stock
  format
  trans_to_qlib
  update
}

main(){
  function_set
  if [ $? -eq 0 ];then
      echo "PIT 执行完成 ！！！"
    else
      echo "PIT 执行失败！！！"
  fi
}

main

