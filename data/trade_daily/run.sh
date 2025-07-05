#/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
echo ${cur_path}

if [ $# -eq 0 ]; then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt=$1
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  else
    echo "参数错误"
    exit 1
fi
if [ $# -eq 0 -o $# -eq 1 ]; then
  echo "dt: "${dt}
 else
   echo "时间区间为：["$dt1" - "$dt2"]"
fi

if [ $# -eq 0 -o $# -eq 1 ]; then
    ${python_path} ${cur_path}/trade_daily.py --date ${dt}
  else
    while [[ $dt1 < $dt2 ]]
      do
        echo "dt1: "${dt1}
        ${python_path} ${cur_path}/trade_daily.py --date ${dt1}
        dt1=`date -d "+1 day $dt1" +%Y-%m-%d`
      done
    echo "dt1: "${dt1}
    ${python_path} ${cur_path}/trade_daily.py --date ${dt1}
fi

# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "执行成功！！！"
  exit 0
 else
  echo "执行失败！！！"
  exit 1
fi
