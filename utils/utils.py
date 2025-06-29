
import yaml

# 配置文件
def get_config():
    with open('./../config.yaml', 'r') as f:
        # 加载配置文件，返回dict
        config = yaml.safe_load(f)
    return config

print(get_config())

