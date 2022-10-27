import time

from benchmarks.evaluation.objective import Objective


class ObjectiveWithAPI(Objective):
    def __init__(self, api) -> None:
        super().__init__(None, None, None)
        self.api = api

    def __call__(self, config):
        _config = config.get_model_for_evaluation()
        start = time.time()
        loss = self.api.eval(_config)
        end = time.time()
        return {
            "loss": self.api.transform(loss),
            "info_dict": {
                "config_id": config.id,
                "val_score": loss,
                "test_score": self.api.test(_config),
                "train_time": end - start,
            },
        }
