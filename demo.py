import hydra
from omegaconf import DictConfig, OmegaConf

from matex.trainers import Trainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
