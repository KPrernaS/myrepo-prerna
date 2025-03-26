import os
import yaml
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = None
        self.request_types = None
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")

        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        if not self.config or "request_types" not in self.config:
            raise ValueError("Invalid config file: 'request_types' key is missing or empty.")

        self.request_types = self.config["request_types"]
        logger.info("Configuration loaded successfully")

    def reload_config(self):
        self.load_config()
        logger.info("Configuration reloaded successfully")

    def update_config(self, new_config):
        if not new_config or "request_types" not in new_config:
            raise ValueError("Invalid configuration: 'request_types' key is required")

        with open(self.config_path, "w") as config_file:
            yaml.safe_dump(new_config, config_file)

        self.reload_config()