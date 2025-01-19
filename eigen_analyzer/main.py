import hydra
from hydra.utils import instantiate

from common.logging import logger as log
from common.setup import initialize
from utils import log_hyperparameters


def train(cfg, device):
    log.debug(f"Instantiating <{cfg.dataset.module._target_}>")
    data_module = instantiate(cfg.dataset.module, cfg)
    data_module.setup()

    log.debug(f"Instantiating the model: <{cfg.task.module._target_}>")
    model = instantiate(cfg.task.module, cfg, device)

    log_hyperparameters(cfg, model)
    log.debug(f"Instantiating <{cfg.trainer.module._target_}>")
    trainer = instantiate(cfg.trainer.module, cfg, device)
    trainer.fit(model, data_module)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    device = initialize(cfg)
    train(cfg, device)


if __name__ == "__main__":
    main()
