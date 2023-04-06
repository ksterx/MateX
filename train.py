import hydra
from omegaconf import DictConfig, OmegaConf

from matex.common import notice
from matex.trainers import Trainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg, logger="mlflow")
    trainer.train()
    trainer.test()
    trainer.play()
    notice.info("Completed!")


if __name__ == "__main__":
    main()
