from matex.common.loggers import MLFlowLogger


def test_mlflow_logger():
    tracking_uri = "http://localhost:5000"
    cfg = {"debug": True}
    logger = MLFlowLogger(tracking_uri=tracking_uri, cfg=cfg)
    logger.log_hparams()
    logger.log_metrics()
