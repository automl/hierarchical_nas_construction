import time

from neps.search_spaces.search_space import SearchSpace

from benchmarks.evaluation.objective import Objective


class ObjectiveWithAPI(Objective):

    def __init__(self, api) -> None:
        super().__init__(None, None, None)
        self.api = api

    def __call__(self, architecture):
        if isinstance(architecture, SearchSpace):
            graph = list(architecture.hyperparameters.values())
            if len(graph) != 1:
                raise Exception(
                    "Only one hyperparameter is allowed for this objective!")
            _config = graph[0].get_model_for_evaluation()
        else:
            _config = architecture
        start = time.time()
        loss = self.api.eval(_config)
        end = time.time()
        return {
            "loss": self.api.transform(loss),
            "info_dict": {
                "config_id": architecture.id,
                "val_score": loss,
                "test_score": self.api.test(_config),
                "train_time": end - start,
            },
        }
