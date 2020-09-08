import yaml
import os


def get_config(path):
    if not os.path.exists(path):
        print("Not exist config path!!!")
        return None
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config