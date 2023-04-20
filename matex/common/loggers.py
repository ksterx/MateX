from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Generator, List, MutableMapping, Optional, Union

from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from matex.common import notice


def flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(
        input_dict: Any, prefixes: List[Optional[str]] = None
    ) -> Generator[Any, Optional[List[str]], List[Any]]:
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


class Logger(ABC):
    @abstractmethod
    def log_hparams(self, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, **kwargs):
        pass


class MLFlowLogger(Logger):
    def __init__(
        self,
        tracking_uri: str,
        cfg: Union[DictConfig, dict],
        exp_name: Optional[str] = None,
    ):
        super().__init__()
        self.client = MlflowClient(tracking_uri)
        self.cfg = cfg

        if cfg["debug"]:
            exp_name = "Debug"
        else:
            exp_name = cfg["exp_name"]

        self.experiment = self.client.get_experiment_by_name(exp_name)
        if self.experiment is None:
            self.experiment_id = self.client.create_experiment(name=exp_name)
            self.experiment = self.client.get_experiment(self.experiment_id)
        else:
            self.experiment_id = self.experiment.experiment_id

        # convert hydra config to dict
        self.run = self.client.create_run(self.experiment_id)
        self.run_id = self.run.info.run_id

        self.local_run_dir = (
            Path(".")
            / Path(tracking_uri.lstrip("file:"))
            / self.experiment_id
            / self.run_id
            / "artifacts"
        ).resolve()

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step: Union[str, int], prefix: str = ""):
        self.client.log_metric(self.run_id, key=prefix + key, value=value, step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: Union[str, int], prefix: str = ""):
        for k, v in metrics.items():
            self.log_metric(k, v, step, prefix)

    def log_hparams(self, params: Union[Dict[str, Any], DictConfig]) -> None:
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
            params = flatten_dict(params)

        for k, v in params.items():
            if len(str(v)) > 250:
                notice.warning(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}"
                )
                continue

            self.log_param(k, v)

    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    def close(self):
        self.client.set_terminated(self.run_id)
