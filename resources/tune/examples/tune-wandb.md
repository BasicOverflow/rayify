# Using Weights & Biases with Tune

---

# Using Weights & Biases with Tune

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=tune-wandb)   




Ray Tune currently offers two lightweight integrations for Weights & Biases. One is the WandbLoggerCallback, which automatically logs metrics reported to Tune to the Wandb API.

The other one is the setup_wandb() function, which can be used with the function API. It automatically initializes the Wandb API with Tune’s training information. You can just use the Wandb API like you would normally do, e.g. using `wandb.log()` to log your training process.

## Running A Weights & Biases Example

In the following example we’re going to use both of the above methods, namely the `WandbLoggerCallback` and the `setup_wandb` function to log metrics.

As the very first step, make sure you’re logged in into wandb on all machines you’re running your training on:
    
    
    wandb login
    

We can then start with a few crucial imports:
    
    
    import numpy as np
    
    import ray
    from ray import tune
    from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
    

Next, let’s define an easy `train_function` function (a Tune `Trainable`) that reports a random loss to Tune. The objective function itself is not important for this example, since we want to focus on the Weights & Biases integration primarily.
    
    
    def train_function(config):
        for i in range(30):
            loss = config["mean"] + config["sd"] * np.random.randn()
            tune.report({"loss": loss})
    

You can define a simple grid-search Tune run using the `WandbLoggerCallback` as follows:
    
    
    def tune_with_callback():
        """Example for using a WandbLoggerCallback with the function API"""
        tuner = tune.Tuner(
            train_function,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            run_config=tune.RunConfig(
                callbacks=[WandbLoggerCallback(project="Wandb_example")]
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
            },
        )
        tuner.fit()
    

To use the `setup_wandb` utility, you simply call this function in your objective. Note that we also use `wandb.log(...)` to log the `loss` to Weights & Biases as a dictionary. Otherwise, this version of our objective is identical to its original.
    
    
    def train_function_wandb(config):
        wandb = setup_wandb(config, project="Wandb_example")
    
        for i in range(30):
            loss = config["mean"] + config["sd"] * np.random.randn()
            tune.report({"loss": loss})
            wandb.log(dict(loss=loss))
    

With the `train_function_wandb` defined, your Tune experiment will set up `wandb` in each trial once it starts!
    
    
    def tune_with_setup():
        """Example for using the setup_wandb utility with the function API"""
        tuner = tune.Tuner(
            train_function_wandb,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
            },
        )
        tuner.fit()
    

Finally, you can also define a class-based Tune `Trainable` by using the `setup_wandb` in the `setup()` method and storing the run object as an attribute. Please note that with the class trainable, you have to pass the trial id, name, and group separately:
    
    
    class WandbTrainable(tune.Trainable):
        def setup(self, config):
            self.wandb = setup_wandb(
                config,
                trial_id=self.trial_id,
                trial_name=self.trial_name,
                group="Example",
                project="Wandb_example",
            )
    
        def step(self):
            for i in range(30):
                loss = self.config["mean"] + self.config["sd"] * np.random.randn()
                self.wandb.log({"loss": loss})
            return {"loss": loss, "done": True}
    
        def save_checkpoint(self, checkpoint_dir: str):
            pass
    
        def load_checkpoint(self, checkpoint_dir: str):
            pass
    

Running Tune with this `WandbTrainable` works exactly the same as with the function API. The below `tune_trainable` function differs from `tune_decorated` above only in the first argument we pass to `Tuner()`:
    
    
    def tune_trainable():
        """Example for using a WandTrainableMixin with the class API"""
        tuner = tune.Tuner(
            WandbTrainable,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
            },
        )
        results = tuner.fit()
    
        return results.get_best_result().config
    

Since you may not have an API key for Wandb, we can _mock_ the Wandb logger and test all three of our training functions as follows. If you are logged in into wandb, you can set `mock_api = False` to actually upload your results to Weights & Biases.
    
    
    import os
    
    mock_api = True
    
    if mock_api:
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_API_KEY", "abcd")
        ray.init(
            runtime_env={"env_vars": {"WANDB_MODE": "disabled", "WANDB_API_KEY": "abcd"}}
        )
    
    tune_with_callback()
    tune_with_setup()
    tune_trainable()
    
    
    
    2022-11-02 16:02:45,355	INFO worker.py:1534 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8266 
    2022-11-02 16:02:46,513	INFO wandb.py:282 -- Already logged into W&B.
    

### Tune Status

Current time:| 2022-11-02 16:03:13  
---|---  
Running for: | 00:00:27.28   
Memory: | 10.8/16.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/3.44 GiB heap, 0.0/1.72 GiB objects 

### Trial Status

Trial name | status | loc |  mean|  sd|  iter|  total time (s)|  loss  
---|---|---|---|---|---|---|---  
train_function_7676d_00000| TERMINATED| 127.0.0.1:14578|  1| 0.411212|  30|  0.236137| 0.828527  
train_function_7676d_00001| TERMINATED| 127.0.0.1:14591|  2| 0.756339|  30|  5.57185 | 3.13156   
train_function_7676d_00002| TERMINATED| 127.0.0.1:14593|  3| 0.436643|  30|  5.50237 | 3.26679   
train_function_7676d_00003| TERMINATED| 127.0.0.1:14595|  4| 0.295929|  30|  5.60986 | 3.70388   
train_function_7676d_00004| TERMINATED| 127.0.0.1:14596|  5| 0.335292|  30|  5.61385 | 4.74294   
  
### Trial Progress

Trial name | date | done | episodes_total | experiment_id | experiment_tag | hostname |  iterations_since_restore|  loss| node_ip |  pid|  time_since_restore|  time_this_iter_s|  time_total_s|  timestamp|  timesteps_since_restore| timesteps_total |  training_iteration| trial_id |  warmup_time  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
train_function_7676d_00000| 2022-11-02_16-02-53| True |  | a9f242fa70184d9dadd8952b16fb0ecc| 0_mean=1,sd=0.4112| Kais-MBP.local.meter|  30| 0.828527| 127.0.0.1| 14578|  0.236137|  0.00381589|  0.236137|  1667430173|  0|  |  30| 7676d_00000|  0.00366998  
train_function_7676d_00001| 2022-11-02_16-03-03| True |  | f57118365bcb4c229fe41c5911f05ad6| 1_mean=2,sd=0.7563| Kais-MBP.local.meter|  30| 3.13156 | 127.0.0.1| 14591|  5.57185 |  0.00627518|  5.57185 |  1667430183|  0|  |  30| 7676d_00001|  0.0027349   
train_function_7676d_00002| 2022-11-02_16-03-03| True |  | 394021d4515d4616bae7126668f73b2b| 2_mean=3,sd=0.4366| Kais-MBP.local.meter|  30| 3.26679 | 127.0.0.1| 14593|  5.50237 |  0.00494576|  5.50237 |  1667430183|  0|  |  30| 7676d_00002|  0.00286222  
train_function_7676d_00003| 2022-11-02_16-03-03| True |  | a575e79c9d95485fa37deaa86267aea4| 3_mean=4,sd=0.2959| Kais-MBP.local.meter|  30| 3.70388 | 127.0.0.1| 14595|  5.60986 |  0.00689816|  5.60986 |  1667430183|  0|  |  30| 7676d_00003|  0.00299597  
train_function_7676d_00004| 2022-11-02_16-03-03| True |  | 91ce57dcdbb54536b1874666b711350d| 4_mean=5,sd=0.3353| Kais-MBP.local.meter|  30| 4.74294 | 127.0.0.1| 14596|  5.61385 |  0.00672579|  5.61385 |  1667430183|  0|  |  30| 7676d_00004|  0.00323987  
      
    
    2022-11-02 16:03:13,913	INFO tune.py:788 -- Total run time: 28.53 seconds (27.28 seconds for the tuning loop).
    

### Tune Status

Current time:| 2022-11-02 16:03:22  
---|---  
Running for: | 00:00:08.49   
Memory: | 9.9/16.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/3.44 GiB heap, 0.0/1.72 GiB objects 

### Trial Status

Trial name | status | loc |  mean|  sd|  iter|  total time (s)|  loss  
---|---|---|---|---|---|---|---  
train_function_wandb_877eb_00000| TERMINATED| 127.0.0.1:14647|  1| 0.738281|  30|  1.61319| 0.555153  
train_function_wandb_877eb_00001| TERMINATED| 127.0.0.1:14660|  2| 0.321178|  30|  1.72447| 2.52109   
train_function_wandb_877eb_00002| TERMINATED| 127.0.0.1:14661|  3| 0.202487|  30|  1.8159 | 2.45412   
train_function_wandb_877eb_00003| TERMINATED| 127.0.0.1:14662|  4| 0.515434|  30|  1.715 | 4.51413   
train_function_wandb_877eb_00004| TERMINATED| 127.0.0.1:14663|  5| 0.216098|  30|  1.72827| 5.2814   
      
    
    (train_function_wandb pid=14647) 2022-11-02 16:03:17,149	INFO wandb.py:282 -- Already logged into W&B.
    

### Trial Progress

Trial name | date | done | episodes_total | experiment_id | experiment_tag | hostname |  iterations_since_restore|  loss| node_ip |  pid|  time_since_restore|  time_this_iter_s|  time_total_s|  timestamp|  timesteps_since_restore| timesteps_total |  training_iteration| trial_id |  warmup_time  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
train_function_wandb_877eb_00000| 2022-11-02_16-03-18| True |  | 7b250c9f31ab484dad1a1fd29823afdf| 0_mean=1,sd=0.7383| Kais-MBP.local.meter|  30| 0.555153| 127.0.0.1| 14647|  1.61319|  0.00232315|  1.61319|  1667430198|  0|  |  30| 877eb_00000|  0.00391102  
train_function_wandb_877eb_00001| 2022-11-02_16-03-22| True |  | 5172868368074557a3044ea3a9146673| 1_mean=2,sd=0.3212| Kais-MBP.local.meter|  30| 2.52109 | 127.0.0.1| 14660|  1.72447|  0.0152011 |  1.72447|  1667430202|  0|  |  30| 877eb_00001|  0.00901699  
train_function_wandb_877eb_00002| 2022-11-02_16-03-22| True |  | b13d9bccb1964b4b95e1a858a3ea64c7| 2_mean=3,sd=0.2025| Kais-MBP.local.meter|  30| 2.45412 | 127.0.0.1| 14661|  1.8159 |  0.00437403|  1.8159 |  1667430202|  0|  |  30| 877eb_00002|  0.00844812  
train_function_wandb_877eb_00003| 2022-11-02_16-03-22| True |  | 869d7ec7a3544a8387985103e626818f| 3_mean=4,sd=0.5154| Kais-MBP.local.meter|  30| 4.51413 | 127.0.0.1| 14662|  1.715 |  0.00247812|  1.715 |  1667430202|  0|  |  30| 877eb_00003|  0.00282907  
train_function_wandb_877eb_00004| 2022-11-02_16-03-22| True |  | 84d3112d66f64325bc469e44b8447ef5| 4_mean=5,sd=0.2161| Kais-MBP.local.meter|  30| 5.2814 | 127.0.0.1| 14663|  1.72827|  0.00517201|  1.72827|  1667430202|  0|  |  30| 877eb_00004|  0.00272107  
      
    
    (train_function_wandb pid=14660) 2022-11-02 16:03:20,600	INFO wandb.py:282 -- Already logged into W&B.
    (train_function_wandb pid=14661) 2022-11-02 16:03:20,600	INFO wandb.py:282 -- Already logged into W&B.
    (train_function_wandb pid=14663) 2022-11-02 16:03:20,628	INFO wandb.py:282 -- Already logged into W&B.
    (train_function_wandb pid=14662) 2022-11-02 16:03:20,723	INFO wandb.py:282 -- Already logged into W&B.
    2022-11-02 16:03:22,565	INFO tune.py:788 -- Total run time: 8.60 seconds (8.48 seconds for the tuning loop).
    

### Tune Status

Current time:| 2022-11-02 16:03:31  
---|---  
Running for: | 00:00:09.28   
Memory: | 9.9/16.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/3.44 GiB heap, 0.0/1.72 GiB objects 

### Trial Status

Trial name | status | loc |  mean|  sd|  iter|  total time (s)|  loss  
---|---|---|---|---|---|---|---  
WandbTrainable_8ca33_00000| TERMINATED| 127.0.0.1:14718|  1| 0.397894|  1|  0.000187159| 0.742345  
WandbTrainable_8ca33_00001| TERMINATED| 127.0.0.1:14737|  2| 0.386883|  1|  0.000151873| 2.5709   
WandbTrainable_8ca33_00002| TERMINATED| 127.0.0.1:14738|  3| 0.290693|  1|  0.00014019 | 2.99601   
WandbTrainable_8ca33_00003| TERMINATED| 127.0.0.1:14739|  4| 0.33333 |  1|  0.00015831 | 3.91276   
WandbTrainable_8ca33_00004| TERMINATED| 127.0.0.1:14740|  5| 0.645479|  1|  0.000150919| 5.47779   
      
    
    (WandbTrainable pid=14718) 2022-11-02 16:03:25,742	INFO wandb.py:282 -- Already logged into W&B.
    

### Trial Progress

Trial name | date | done | episodes_total | experiment_id | hostname |  iterations_since_restore|  loss| node_ip |  pid|  time_since_restore|  time_this_iter_s|  time_total_s|  timestamp|  timesteps_since_restore| timesteps_total |  training_iteration| trial_id |  warmup_time  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
WandbTrainable_8ca33_00000| 2022-11-02_16-03-27| True |  | 3adb4d0ae0d74d1c9ddd07924b5653b0| Kais-MBP.local.meter|  1| 0.742345| 127.0.0.1| 14718|  0.000187159|  0.000187159|  0.000187159|  1667430207|  0|  |  1| 8ca33_00000|  1.31382  
WandbTrainable_8ca33_00001| 2022-11-02_16-03-31| True |  | f1511cfd51f94b3d9cf192181ccc08a9| Kais-MBP.local.meter|  1| 2.5709 | 127.0.0.1| 14737|  0.000151873|  0.000151873|  0.000151873|  1667430211|  0|  |  1| 8ca33_00001|  1.31668  
WandbTrainable_8ca33_00002| 2022-11-02_16-03-31| True |  | a7528ec6adf74de0b73aa98ebedab66d| Kais-MBP.local.meter|  1| 2.99601 | 127.0.0.1| 14738|  0.00014019 |  0.00014019 |  0.00014019 |  1667430211|  0|  |  1| 8ca33_00002|  1.32008  
WandbTrainable_8ca33_00003| 2022-11-02_16-03-31| True |  | b7af756ca586449ba2d4c44141b53b06| Kais-MBP.local.meter|  1| 3.91276 | 127.0.0.1| 14739|  0.00015831 |  0.00015831 |  0.00015831 |  1667430211|  0|  |  1| 8ca33_00003|  1.31879  
WandbTrainable_8ca33_00004| 2022-11-02_16-03-31| True |  | 196624f42bcc45c18a26778573a43a2c| Kais-MBP.local.meter|  1| 5.47779 | 127.0.0.1| 14740|  0.000150919|  0.000150919|  0.000150919|  1667430211|  0|  |  1| 8ca33_00004|  1.31945  
      
    
    (WandbTrainable pid=14739) 2022-11-02 16:03:30,360	INFO wandb.py:282 -- Already logged into W&B.
    (WandbTrainable pid=14740) 2022-11-02 16:03:30,393	INFO wandb.py:282 -- Already logged into W&B.
    (WandbTrainable pid=14737) 2022-11-02 16:03:30,454	INFO wandb.py:282 -- Already logged into W&B.
    (WandbTrainable pid=14738) 2022-11-02 16:03:30,510	INFO wandb.py:282 -- Already logged into W&B.
    2022-11-02 16:03:31,985	INFO tune.py:788 -- Total run time: 9.40 seconds (9.27 seconds for the tuning loop).
    
    
    
    {'mean': 1, 'sd': 0.3978937765393781, 'wandb': {'project': 'Wandb_example'}}
    

This completes our Tune and Wandb walk-through. In the following sections you can find more details on the API of the Tune-Wandb integration.

## Tune Wandb API Reference

### WandbLoggerCallback

_class _ray.air.integrations.wandb.WandbLoggerCallback(_project : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _group : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _api_key_file : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _api_key : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _excludes : [List](https://docs.python.org/3/library/typing.html#typing.List "\(in Python v3.14\)")[[str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")] | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _log_config : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = False_, _upload_checkpoints : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = False_, _save_checkpoints : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = False_, _upload_timeout : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") = 1800_, _video_kwargs : [dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _image_kwargs : [dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _** kwargs_)[[source]](../../_modules/ray/air/integrations/wandb.html#WandbLoggerCallback)
    


Example
    
    
    import random
    
    from ray import tune
    from ray.air.integrations.wandb import WandbLoggerCallback
    
    
    def train_func(config):
        offset = random.random() / 5
        for epoch in range(2, config["epochs"]):
            acc = 1 - (2 + config["lr"]) ** -epoch - random.random() / epoch - offset
            loss = (2 + config["lr"]) ** -epoch + random.random() / epoch + offset
            train.report({"acc": acc, "loss": loss})
    
    
    tuner = tune.Tuner(
        train_func,
        param_space={
            "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
            "epochs": 10,
        },
        run_config=tune.RunConfig(
            callbacks=[WandbLoggerCallback(project="Optimization_Project")]
        ),
    )
    results = tuner.fit()
    

Parameters:
    

  * **project** – Name of the Wandb project. Mandatory.

  * **group** – Name of the Wandb group. Defaults to the trainable name.

  * **api_key_file** – Path to file containing the Wandb API KEY. This file only needs to be present on the node running the Tune script if using the WandbLogger.

  * **api_key** – Wandb API Key. Alternative to setting `api_key_file`.

  * **excludes** – List of metrics and config that should be excluded from the log.

  * **log_config** – Boolean indicating if the `config` parameter of the `results` dict should be logged. This makes sense if parameters will change during training, e.g. with PopulationBasedTraining. Defaults to False.

  * **upload_checkpoints** – If `True`, model checkpoints will be uploaded to Wandb as artifacts. Defaults to `False`.

  * **video_kwargs** – Dictionary of keyword arguments passed to wandb.Video() when logging videos. Videos have to be logged as 5D numpy arrays to be affected by this parameter. For valid keyword arguments, see <https://docs.wandb.ai/ref/python/data-types/video/>. Defaults to `None`.

  * **image_kwargs** – Dictionary of keyword arguments passed to wandb.Image() when logging images. Images have to be logged as 3D or 4D numpy arrays to be affected by this parameter. For valid keyword arguments, see <https://docs.wandb.ai/ref/python/data-types/image/>. Defaults to `None`.

  * ****kwargs** – The keyword arguments will be passed to `wandb.init()`.

Wandb’s `group`, `run_id` and `run_name` are automatically selected by Tune, but can be overwritten by filling out the respective configuration values.

Please see here for all other valid configuration settings: <https://docs.wandb.ai/ref/python/init/>

**PublicAPI (alpha):** This API is in alpha and may change before becoming stable.

### setup_wandb

ray.air.integrations.wandb.setup_wandb(_config : [Dict](https://docs.python.org/3/library/typing.html#typing.Dict "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _api_key : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _api_key_file : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _rank_zero_only : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = True_, _** kwargs_) → wandb.wandb_run.Run | wandb.sdk.lib.disabled.RunDisabled[[source]](../../_modules/ray/air/integrations/wandb.html#setup_wandb)
    

Set up a Weights & Biases session.

This function can be used to initialize a Weights & Biases session in a (distributed) training or tuning run.

By default, the run ID is the trial ID, the run name is the trial name, and the run group is the experiment name. These settings can be overwritten by passing the respective arguments as `kwargs`, which will be passed to `wandb.init()`.

In distributed training with Ray Train, only the zero-rank worker will initialize wandb. All other workers will return a disabled run object, so that logging is not duplicated in a distributed run. This can be disabled by passing `rank_zero_only=False`, which will then initialize wandb in every training worker.

The `config` argument will be passed to Weights and Biases and will be logged as the run configuration.

If no API key or key file are passed, wandb will try to authenticate using locally stored credentials, created for instance by running `wandb login`.

Keyword arguments passed to `setup_wandb()` will be passed to `wandb.init()` and take precedence over any potential default settings.

Parameters:
    

  * **config** – Configuration dict to be logged to Weights and Biases. Can contain arguments for `wandb.init()` as well as authentication information.

  * **api_key** – API key to use for authentication with Weights and Biases.

  * **api_key_file** – File pointing to API key for with Weights and Biases.

  * **rank_zero_only** – If True, will return an initialized session only for the rank 0 worker in distributed training. If False, will initialize a session for all workers.

  * **kwargs** – Passed to `wandb.init()`.

Example
    
    
    from ray.air.integrations.wandb import setup_wandb
    
    def training_loop(config):
        wandb = setup_wandb(config)
        # ...
        wandb.log({"loss": 0.123})
    

**PublicAPI (alpha):** This API is in alpha and may change before becoming stable.