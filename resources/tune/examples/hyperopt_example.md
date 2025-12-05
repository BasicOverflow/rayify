# Running Tune experiments with HyperOpt

---

# Running Tune experiments with HyperOpt

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=ray-tune-hyperopt_example)   

In this tutorial we introduce HyperOpt, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with HyperOpt and, as a result, allow you to seamlessly scale up a Hyperopt optimization process - without sacrificing performance.

HyperOpt provides gradient/derivative-free optimization able to handle noise over the objective landscape, including evolutionary, bandit, and Bayesian optimization algorithms. HyperOpt internally supports search spaces which are continuous, discrete or a mixture of thereof. It also provides a library of functions on which to test the optimization algorithms and compare with other benchmarks.

In this example we minimize a simple objective to briefly demonstrate the usage of HyperOpt with Ray Tune via `HyperOptSearch`. It’s useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `hyperopt==0.2.5` library is installed. To learn more, please refer to [HyperOpt website](http://hyperopt.github.io/hyperopt).

We include a important example on conditional search spaces (stringing together relationships among hyperparameters).

Background information:

  * [HyperOpt website](http://hyperopt.github.io/hyperopt)

Necessary requirements:

  * `pip install "ray[tune]" hyperopt==0.2.5`

Show code cell content Hide code cell content
    
    
    # install in a hidden cell
    # !pip install "ray[tune]"
    !pip install hyperopt==0.2.5
    
    
    
    Requirement already satisfied: hyperopt==0.2.5 in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (0.2.5)
    Requirement already satisfied: numpy in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (2.2.3)
    Requirement already satisfied: scipy in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (1.15.2)
    Requirement already satisfied: six in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (1.17.0)
    Requirement already satisfied: networkx>=2.2 in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (3.4.2)
    Requirement already satisfied: future in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (1.0.0)
    Requirement already satisfied: tqdm in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (4.67.1)
    Requirement already satisfied: cloudpickle in /opt/miniconda3/envs/hyperopt_example/lib/python3.11/site-packages (from hyperopt==0.2.5) (3.1.1)
    

Click below to see all the imports we need for this example.

Show code cell source Hide code cell source
    
    
    import time
    
    import ray
    from ray import tune
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.hyperopt import HyperOptSearch
    from hyperopt import hp
    

Let’s start by defining a simple evaluation function. We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment. This setup assumes that we’re running multiple `step`s of an experiment and try to tune two hyperparameters, namely `width` and `height`.
    
    
    def evaluate(step, width, height):
        time.sleep(0.1)
        return (0.1 + width * step / 100) ** (-1) + height * 0.1
    

Next, our `objective` function takes a Tune `config`, evaluates the `score` of your experiment in a training loop, and uses `tune.report` to report the `score` back to Tune.
    
    
    def objective(config):
        for step in range(config["steps"]):
            score = evaluate(step, config["width"], config["height"])
            tune.report({"iterations": step, "mean_loss": score})
    
    
    
    ray.init(configure_logging=False)
    

While defining the search algorithm, we may choose to provide an initial set of hyperparameters that we believe are especially promising or informative, and pass this information as a helpful starting point for the `HyperOptSearch` object.

We also set the maximum concurrent trials to `4` with a `ConcurrencyLimiter`.
    
    
    initial_params = [
        {"width": 1, "height": 2, "activation": "relu"},
        {"width": 4, "height": 2, "activation": "tanh"},
    ]
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    

The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples. (you can decrease this if it takes too long on your machine).
    
    
    num_samples = 1000
    

Next we define a search space. The critical assumption is that the optimal hyperparameters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.
    
    
    search_config = {
        "steps": 100,
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
        "activation": tune.choice(["relu", "tanh"])
    }
    

Finally, we run the experiment to `"min"`imize the “mean_loss” of the `objective` by searching `search_config` via `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=search_config,
    )
    results = tuner.fit()
    

### Tune Status

Current time:| 2025-02-18 13:14:59  
---|---  
Running for: | 00:00:36.03   
Memory: | 22.1/36.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs 

### Trial Status

Trial name | status | loc | activation |  height|  steps|  width|  loss|  iter|  total time (s)|  iterations  
---|---|---|---|---|---|---|---|---|---|---  
objective_5b05c00a| TERMINATED| 127.0.0.1:50205| relu |  2 |  100|  1 |  1.11743 |  100|  10.335 |  99  
objective_b813f49d| TERMINATED| 127.0.0.1:50207| tanh |  2 |  100|  4 |  0.446305|  100|  10.3299|  99  
objective_6dadd2bd| TERMINATED| 127.0.0.1:50212| tanh | -40.9318|  100|  6.20615 | -3.93303 |  100|  10.3213|  99  
objective_9faffc0f| TERMINATED| 127.0.0.1:50217| tanh |  91.9688|  100|  9.25147 |  9.30488 |  100|  10.353 |  99  
objective_7834e74c| TERMINATED| 127.0.0.1:50266| tanh | -17.9521|  100| 11.436 | -1.70766 |  100|  10.3753|  99  
objective_741253c7| TERMINATED| 127.0.0.1:50271| tanh |  58.1279|  100|  0.737879|  7.01689 |  100|  10.3565|  99  
objective_39682bcf| TERMINATED| 127.0.0.1:50272| tanh | -31.2589|  100|  4.89265 | -2.92361 |  100|  10.3225|  99  
objective_bfc7e150| TERMINATED| 127.0.0.1:50274| tanh | -14.7877|  100|  7.36477 | -1.34347 |  100|  10.3744|  99  
objective_e1f1a193| TERMINATED| 127.0.0.1:50314| tanh |  50.9579|  100| 17.4499 |  5.15334 |  100|  10.3675|  99  
objective_1192dd4e| TERMINATED| 127.0.0.1:50316| tanh |  66.5306|  100| 16.9549 |  6.71228 |  100|  10.3478|  99  
  
Here are the hyperparameters found to minimize the mean loss of the defined objective.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'steps': 100, 'width': 6.206149011253133, 'height': -40.93182668460948, 'activation': 'tanh'}
    

## Conditional search spaces

Sometimes we may want to build a more complicated search space that has conditional dependencies on other hyperparameters. In this case, we pass a nested dictionary to `objective_two`, which has been slightly adjusted from `objective` to deal with the conditional search space.
    
    
    def evaluation_fn(step, width, height, mult=1):
        return (0.1 + width * step / 100) ** (-1) + height * 0.1 * mult
    
    
    
    def objective_two(config):
        width, height = config["width"], config["height"]
        sub_dict = config["activation"]
        mult = sub_dict.get("mult", 1)
        
        for step in range(config["steps"]):
            intermediate_score = evaluation_fn(step, width, height, mult)
            tune.report({"iterations": step, "mean_loss": intermediate_score})
            time.sleep(0.1)
    
    
    
    conditional_space = {
        "activation": hp.choice(
            "activation",
            [
                {"activation": "relu", "mult": hp.uniform("mult", 1, 2)},
                {"activation": "tanh"},
            ],
        ),
        "width": hp.uniform("width", 0, 20),
        "height": hp.uniform("height", -100, 100),
        "steps": 100,
    }
    

Now we the define the search algorithm built from `HyperOptSearch` constrained by `ConcurrencyLimiter`. When the hyperparameter search space is conditional, we pass it (`conditional_space`) into `HyperOptSearch`.
    
    
    algo = HyperOptSearch(space=conditional_space, metric="mean_loss", mode="min")
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    

Now we run the experiment, this time with an empty `config` because we instead provided `space` to the `HyperOptSearch` `search_alg`.
    
    
    tuner = tune.Tuner(
        objective_two,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    

### Tune Status

Current time:| 2025-02-18 13:15:34  
---|---  
Running for: | 00:00:34.71   
Memory: | 22.9/36.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs 

### Trial Status

Trial name | status | loc | activation/activatio n |  activation/mult|  height|  steps|  width|  loss|  iter|  total time (s)|  iterations  
---|---|---|---|---|---|---|---|---|---|---|---  
objective_two_13de5867| TERMINATED| 127.0.0.1:50350| tanh|  | -35.0329 |  100| 13.2254 |  -3.42749 |  100|  10.2102|  99  
objective_two_3100f5ee| TERMINATED| 127.0.0.1:50355| relu|  1.44584|  76.2581 |  100|  0.123165|  15.5316 |  100|  10.2683|  99  
objective_two_6b4044aa| TERMINATED| 127.0.0.1:50356| relu|  1.67475|  57.9612 |  100| 19.4794 |  9.75866 |  100|  10.2724|  99  
objective_two_4aa1269b| TERMINATED| 127.0.0.1:50357| tanh|  |  -9.95686|  100| 12.9749 |  -0.918437|  100|  10.2373|  99  
objective_two_d8c42c9f| TERMINATED| 127.0.0.1:50402| tanh|  | -96.6184 |  100|  8.03869 |  -9.53774 |  100|  10.2407|  99  
objective_two_de956d10| TERMINATED| 127.0.0.1:50404| relu|  1.06986|  9.64996|  100|  0.672962|  2.3375 |  100|  10.2427|  99  
objective_two_0e2b2751| TERMINATED| 127.0.0.1:50413| tanh|  | -82.9292 |  100| 17.4889 |  -8.2355 |  100|  10.293 |  99  
objective_two_dad93a03| TERMINATED| 127.0.0.1:50415| relu|  1.85364| -63.6309 |  100| 16.6414 | -11.7345 |  100|  10.285 |  99  
objective_two_e472727a| TERMINATED| 127.0.0.1:50442| relu|  1.2359 | -75.2253 |  100|  7.49782 |  -9.16415 |  100|  10.2506|  99  
objective_two_5d1eacff| TERMINATED| 127.0.0.1:50449| tanh|  | -20.158 |  100|  6.18643 |  -1.85514 |  100|  10.2566|  99  
  
Finally, we again show the hyperparameters that minimize the mean loss defined by the score of the objective function above.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'activation': {'activation': 'relu', 'mult': 1.8536380640438768}, 'height': -63.630920754630125, 'steps': 100, 'width': 16.641403933591928}