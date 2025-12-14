
## 检查local_qlib:latest是否存在
修改env.py中 DockerEnv 类的prepare 方法， 替换 /root/anaconda3/envs/rdagent/lib/python3.10/site-packages/rdagent/utils/env.py

## 使用自定义数据
修改 factor_data_template 
    generate.py 中的provider_uri 、日期参数、字段获取逻辑
    README.md: 字段解释信息
替换 /root/anaconda3/envs/rdagent/lib/python3.10/site-packages/rdagent/scenarios/qlib/experiment/factor_data_template/

## 因子生成。替换回测数据源及日期
修改 factor_template 中模版信息。替换 /root/anaconda3/envs/rdagent/lib/python3.10/site-packages/rdagent/scenarios/qlib/experiment/factor_template 下模版



