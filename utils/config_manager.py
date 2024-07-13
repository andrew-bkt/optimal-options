# utils/config_manager.py
import yaml

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_nested(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
