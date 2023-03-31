import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

METRICS_LIST = ["loss", "acc", "top_3_acc", "top_5_acc", "top_10_acc"]