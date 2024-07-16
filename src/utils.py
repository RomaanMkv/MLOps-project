from hydra import initialize, compose


CONFIG_PATH = "../configs"
CONFIG_NAME = "main"


def init_hydra():
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(config_name=CONFIG_NAME)
        return cfg
    