#/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
echo ${cur_path}

if [ $# -eq 1 ]; then
   rolling_step=$1
  else
   rolling_step=10
fi
echo "rolling_step: "${rolling_step}

${python_path} ${cur_path}/train_scheduler.py routine --rolling_step ${rolling_step}

if [ $? -eq 0 ]; then
  echo "执行成功！"
else
  echo "执行失败！"
  exit 1
fi

