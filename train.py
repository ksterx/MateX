import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from matex.common import notice
from matex.trainers import Trainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Ray initialization
    ray.init()
    resources = ray.available_resources()
    notice.info(f"Available CPUs: {resources['CPU']}")
    try:
        notice.info(f"Available GPUs: {resources['GPU']}")
    except KeyError:
        notice.warning("No GPUs available!")

    trainer = Trainer(cfg, logger="mlflow")
    trainer.train()
    trainer.test()
    trainer.play()
    notice.info("Completed!")


if __name__ == "__main__":
    main()
