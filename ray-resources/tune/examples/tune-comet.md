# Using Comet with Tune

---

# Using Comet with Tune

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=tune-comet)   

[Comet](https://www.comet.ml/site/) is a tool to manage and optimize the entire ML lifecycle, from experiment tracking, model optimization and dataset versioning to model production monitoring.



## Example

To illustrate logging your trial results to Comet, we’ll define a simple training function that simulates a `loss` metric:
    
    
    import numpy as np
    from ray import tune
    
    
    def train_function(config):
        for i in range(30):
            loss = config["mean"] + config["sd"] * np.random.randn()
            tune.report({"loss": loss})
    

Now, given that you provide your Comet API key and your project name like so:
    
    
    api_key = "YOUR_COMET_API_KEY"
    project_name = "YOUR_COMET_PROJECT_NAME"
    

You can add a Comet logger by specifying the `callbacks` argument in your `RunConfig()` accordingly:
    
    
    from ray.air.integrations.comet import CometLoggerCallback
    
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        run_config=tune.RunConfig(
            callbacks=[
                CometLoggerCallback(
                    api_key=api_key, project_name=project_name, tags=["comet_example"]
                )
            ],
        ),
        param_space={"mean": tune.grid_search([1, 2, 3]), "sd": tune.uniform(0.2, 0.8)},
    )
    results = tuner.fit()
    
    print(results.get_best_result().config)
    

## Tune Comet Logger

Ray Tune offers an integration with Comet through the `CometLoggerCallback`, which automatically logs metrics and parameters reported to Tune to the Comet UI.

Click on the following dropdown to see this callback API in detail:

_class _ray.air.integrations.comet.CometLoggerCallback(_online : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = True_, _tags : [List](https://docs.python.org/3/library/typing.html#typing.List "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")] | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _save_checkpoints : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = False_, _** experiment_kwargs_)[[source]](../../_modules/ray/air/integrations/comet.html#CometLoggerCallback)
    

CometLoggerCallback for logging Tune results to Comet.

Comet (<https://comet.ml/site/>) is a tool to manage and optimize the entire ML lifecycle, from experiment tracking, model optimization and dataset versioning to model production monitoring.

This Ray Tune `LoggerCallback` sends metrics and parameters to Comet for tracking.

In order to use the CometLoggerCallback you must first install Comet via `pip install comet_ml`

Then set the following environment variables `export COMET_API_KEY=<Your API Key>`

Alternatively, you can also pass in your API Key as an argument to the CometLoggerCallback constructor.

`CometLoggerCallback(api_key=<Your API Key>)`

Parameters:
    

  * **online** – Whether to make use of an Online or Offline Experiment. Defaults to True.

  * **tags** – Tags to add to the logged Experiment. Defaults to None.

  * **save_checkpoints** – If `True`, model checkpoints will be saved to Comet ML as artifacts. Defaults to `False`.

  * ****experiment_kwargs** – Other keyword arguments will be passed to the constructor for comet_ml.Experiment (or OfflineExperiment if online=False).

Please consult the Comet ML documentation for more information on the Experiment and OfflineExperiment classes: <https://comet.ml/site/>

Example:
    
    
    from ray.air.integrations.comet import CometLoggerCallback
    tune.run(
        train,
        config=config
        callbacks=[CometLoggerCallback(
            True,
            ['tag1', 'tag2'],
            workspace='my_workspace',
            project_name='my_project_name'
            )]
    )