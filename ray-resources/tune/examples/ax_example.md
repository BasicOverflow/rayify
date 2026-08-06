# Running Tune experiments with AxSearch

---

# Running Tune experiments with AxSearch

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=ray-tune-ax_example)   

In this tutorial we introduce Ax, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Ax and, as a result, allow you to seamlessly scale up a Ax optimization process - without sacrificing performance.

Ax is a platform for optimizing any kind of experiment, including machine learning experiments, A/B tests, and simulations. Ax can optimize discrete configurations (e.g., variants of an A/B test) using multi-armed bandit optimization, and continuous/ordered configurations (e.g. float/int parameters) using Bayesian optimization. Results of A/B tests and simulations with reinforcement learning agents often exhibit high amounts of noise. Ax supports state-of-the-art algorithms which work better than traditional Bayesian optimization in high-noise settings. Ax also supports multi-objective and constrained optimization which are common to real-world problems (e.g. improving load time without increasing data use). Ax belongs to the domain of “derivative-free” and “black-box” optimization.

In this example we minimize a simple objective to briefly demonstrate the usage of AxSearch with Ray Tune via `AxSearch`. It’s useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `ax-platform==0.2.4` library is installed withe python version >= 3.7. To learn more, please refer to the [Ax website](https://ax.dev/).

Click below to see all the imports we need for this example.

Show code cell source Hide code cell source
    
    
    import numpy as np
    import time
    
    import ray
    from ray import tune
    from ray.tune.search.ax import AxSearch
    

Let’s start by defining a classic benchmark for global optimization. The form here is explicit for demonstration, yet it is typically a black-box. We artificially sleep for a bit (`0.02` seconds) to simulate a long-running ML experiment. This setup assumes that we’re running multiple `step`s of an experiment and try to tune 6-dimensions of the `x` hyperparameter.
    
    
    def landscape(x):
        """
        Hartmann 6D function containing 6 local minima.
        It is a classic benchmark for developing global optimization algorithms.
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return y
    

Next, our `objective` function takes a Tune `config`, evaluates the `landscape` of our experiment in a training loop, and uses `tune.report` to report the `landscape` back to Tune.
    
    
    def objective(config):
        for i in range(config["iterations"]):
            x = np.array([config.get("x{}".format(i + 1)) for i in range(6)])
            tune.report(
                {"timesteps_total": i, "landscape": landscape(x), "l2norm": np.sqrt((x ** 2).sum())}
            )
            time.sleep(0.02)
    

Next we define a search space. The critical assumption is that the optimal hyperparameters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.
    
    
    search_space = {
        "iterations":100,
        "x1": tune.uniform(0.0, 1.0),
        "x2": tune.uniform(0.0, 1.0),
        "x3": tune.uniform(0.0, 1.0),
        "x4": tune.uniform(0.0, 1.0),
        "x5": tune.uniform(0.0, 1.0),
        "x6": tune.uniform(0.0, 1.0)
    }
    

Now we define the search algorithm from `AxSearch`. If you want to constrain your parameters or even the space of outcomes, that can be easily done by passing the argumentsas below.
    
    
    algo = AxSearch(
        parameter_constraints=["x1 + x2 <= 2.0"],
        outcome_constraints=["l2norm <= 1.25"],
    )
    

We also use `ConcurrencyLimiter` to constrain to 4 concurrent trials.# Running Tune experiments with AxSearch#

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=ray-tune-ax_example)   

In this tutorial we introduce Ax, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Ax and, as a result, allow you to seamlessly scale up a Ax optimization process - without sacrificing performance.

Ax is a platform for optimizing any kind of experiment, including machine learning experiments, A/B tests, and simulations. Ax can optimize discrete configurations (e.g., variants of an A/B test) using multi-armed bandit optimization, and continuous/ordered configurations (e.g. float/int parameters) using Bayesian optimization. Results of A/B tests and simulations with reinforcement learning agents often exhibit high amounts of noise. Ax supports state-of-the-art algorithms which work better than traditional Bayesian optimization in high-noise settings. Ax also supports multi-objective and constrained optimization which are common to real-world problems (e.g. improving load time without increasing data use). Ax belongs to the domain of “derivative-free” and “black-box” optimization.

In this example we minimize a simple objective to briefly demonstrate the usage of AxSearch with Ray Tune via `AxSearch`. It’s useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `ax-platform==0.2.4` library is installed withe python version >= 3.7. To learn more, please refer to the [Ax website](https://ax.dev/).

Click below to see all the imports we need for this example.

Show code cell source Hide code cell source
    
    
    import numpy as np
    import time
    
    import ray
    from ray import tune
    from ray.tune.search.ax import AxSearch
    

Let’s start by defining a classic benchmark for global optimization. The form here is explicit for demonstration, yet it is typically a black-box. We artificially sleep for a bit (`0.02` seconds) to simulate a long-running ML experiment. This setup assumes that we’re running multiple `step`s of an experiment and try to tune 6-dimensions of the `x` hyperparameter.
    
    
    def landscape(x):
        """
        Hartmann 6D function containing 6 local minima.
        It is a classic benchmark for developing global optimization algorithms.
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return y
    

Next, our `objective` function takes a Tune `config`, evaluates the `landscape` of our experiment in a training loop, and uses `tune.report` to report the `landscape` back to Tune.
    
    
    def objective(config):
        for i in range(config["iterations"]):
            x = np.array([config.get("x{}".format(i + 1)) for i in range(6)])
            tune.report(
                {"timesteps_total": i, "landscape": landscape(x), "l2norm": np.sqrt((x ** 2).sum())}
            )
            time.sleep(0.02)
    

Next we define a search space. The critical assumption is that the optimal hyperparameters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.
    
    
    search_space = {
        "iterations":100,
        "x1": tune.uniform(0.0, 1.0),
        "x2": tune.uniform(0.0, 1.0),
        "x3": tune.uniform(0.0, 1.0),
        "x4": tune.uniform(0.0, 1.0),
        "x5": tune.uniform(0.0, 1.0),
        "x6": tune.uniform(0.0, 1.0)
    }
    

Now we define the search algorithm from `AxSearch`. If you want to constrain your parameters or even the space of outcomes, that can be easily done by passing the argumentsas below.
    
    
    algo = AxSearch(
        parameter_constraints=["x1 + x2 <= 2.0"],
        outcome_constraints=["l2norm <= 1.25"],
    )
    

We also use `ConcurrencyLimiter` to constrain to 4 concurrent trials.
    
    
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
    

The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples. You can decrease this if it takes too long on your machine, or you can set a time limit easily through `stop` argument in the `RunConfig()` as we will show here.
    
    
    num_samples = 100
    stop_timesteps = 200
    

Finally, we run the experiment to find the global minimum of the provided landscape (which contains 5 false minima). The argument to metric, `"landscape"`, is provided via the `objective` function’s `tune.report`. The experiment `"min"`imizes the “mean_loss” of the `landscape` by searching within `search_space` via `algo`, `num_samples` times or when `"timesteps_total": stop_timesteps`. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="landscape",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        run_config=tune.RunConfig(
            name="ax",
            stop={"timesteps_total": stop_timesteps}
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    
    
    [INFO 07-22 15:04:18] ax.service.ax_client: Starting optimization with verbose logging. To disable logging, set the `verbose_logging` argument to `False`. Note that float values in the logs are rounded to 6 decimal points.
    [INFO 07-22 15:04:18] ax.service.utils.instantiation: Created search space: SearchSpace(parameters=[FixedParameter(name='iterations', parameter_type=INT, value=100), RangeParameter(name='x1', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x2', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x3', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x4', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x5', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x6', parameter_type=FLOAT, range=[0.0, 1.0])], parameter_constraints=[ParameterConstraint(1.0*x1 + 1.0*x2 <= 2.0)]).
    [INFO 07-22 15:04:18] ax.modelbridge.dispatch_utils: Using Bayesian optimization since there are more ordered parameters than there are categories for the unordered categorical parameters.
    [INFO 07-22 15:04:18] ax.modelbridge.dispatch_utils: Using Bayesian Optimization generation strategy: GenerationStrategy(name='Sobol+GPEI', steps=[Sobol for 12 trials, GPEI for subsequent trials]). Iterations after 12 will take longer to generate due to  model-fitting.
    Detected sequential enforcement. Be sure to use a ConcurrencyLimiter.
    

== Status ==  
Current time: 2022-07-22 15:04:35 (running for 00:00:16.56)  
Memory usage on this node: 9.9/16.0 GiB  
Using FIFO scheduling algorithm.  
Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/5.13 GiB heap, 0.0/2.0 GiB objects  
Current best trial: 34b7abda with landscape=-1.6624439263544026 and parameters={'iterations': 100, 'x1': 0.26526361983269453, 'x2': 0.9248840995132923, 'x3': 0.15171580761671066, 'x4': 0.43602637108415365, 'x5': 0.8573104059323668, 'x6': 0.08981018699705601}  
Result logdir: /Users/kai/ray_results/ax  
Number of trials: 10/10 (10 TERMINATED)  
Trial name | status | loc |  iterations|  x1|  x2|  x3|  x4|  x5|  x6|  iter|  total time (s)|  ts|  landscape|  l2norm  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
objective_2dfbe86a| TERMINATED| 127.0.0.1:44721|  100| 0.0558336| 0.0896192| 0.958956 | 0.234474 | 0.174516 | 0.970311 |  100|  2.57372|  99| -0.805233 |  1.39917  
objective_2fa776c0| TERMINATED| 127.0.0.1:44726|  100| 0.744772 | 0.754537 | 0.0950125| 0.273877 | 0.0966829| 0.368943 |  100|  2.6361 |  99| -0.11286 |  1.16341  
objective_2fabaa1a| TERMINATED| 127.0.0.1:44727|  100| 0.405704 | 0.374626 | 0.935628 | 0.222185 | 0.787212 | 0.00812439|  100|  2.62393|  99| -0.11348 |  1.35995  
objective_2faee7c0| TERMINATED| 127.0.0.1:44728|  100| 0.664728 | 0.207519 | 0.359514 | 0.704578 | 0.755882 | 0.812402 |  100|  2.62069|  99| -0.0119837 |  1.53035  
objective_313d3d3a| TERMINATED| 127.0.0.1:44747|  100| 0.0418746| 0.992783 | 0.906027 | 0.594429 | 0.825393 | 0.646362 |  100|  3.16233|  99| -0.00677976|  1.80573  
objective_32c9acd8| TERMINATED| 127.0.0.1:44726|  100| 0.126064 | 0.703408 | 0.344681 | 0.337363 | 0.401396 | 0.679202 |  100|  3.12119|  99| -0.904622 |  1.16864  
objective_32cf8ca2| TERMINATED| 127.0.0.1:44756|  100| 0.0910936| 0.304138 | 0.869848 | 0.405435 | 0.567922 | 0.228608 |  100|  2.70791|  99| -0.146532 |  1.18178  
objective_32d8dd20| TERMINATED| 127.0.0.1:44758|  100| 0.603178 | 0.409057 | 0.729056 | 0.0825984| 0.572948 | 0.508304 |  100|  2.64158|  99| -0.247223 |  1.28691  
objective_34adf04a| TERMINATED| 127.0.0.1:44768|  100| 0.454189 | 0.271772 | 0.530871 | 0.991841 | 0.691843 | 0.472366 |  100|  2.70327|  99| -0.0132915 |  1.49917  
objective_34b7abda| TERMINATED| 127.0.0.1:44771|  100| 0.265264 | 0.924884 | 0.151716 | 0.436026 | 0.85731 | 0.0898102 |  100|  2.68521|  99| -1.66244 |  1.37185  
  
  

    
    
    [INFO 07-22 15:04:19] ax.service.ax_client: Generated new trial 0 with parameters {'x1': 0.055834, 'x2': 0.089619, 'x3': 0.958956, 'x4': 0.234474, 'x5': 0.174516, 'x6': 0.970311, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 1 with parameters {'x1': 0.744772, 'x2': 0.754537, 'x3': 0.095012, 'x4': 0.273877, 'x5': 0.096683, 'x6': 0.368943, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 2 with parameters {'x1': 0.405704, 'x2': 0.374626, 'x3': 0.935628, 'x4': 0.222185, 'x5': 0.787212, 'x6': 0.008124, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 3 with parameters {'x1': 0.664728, 'x2': 0.207519, 'x3': 0.359514, 'x4': 0.704578, 'x5': 0.755882, 'x6': 0.812402, 'iterations': 100}.
    
    
    
    Result for objective_2dfbe86a:
      date: 2022-07-22_15-04-22
      done: false
      experiment_id: 4ef8a12ac94a4f4fa483ec18e347967f
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3991721132671366
      landscape: -0.8052333562869153
      node_ip: 127.0.0.1
      pid: 44721
      time_since_restore: 0.00022912025451660156
      time_this_iter_s: 0.00022912025451660156
      time_total_s: 0.00022912025451660156
      timestamp: 1658498662
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2dfbe86a
      warmup_time: 0.0035619735717773438
      
    
    
    
    [INFO 07-22 15:04:24] ax.service.ax_client: Completed trial 0 with data: {'landscape': (-0.805233, None), 'l2norm': (1.399172, None)}.
    [INFO 07-22 15:04:24] ax.service.ax_client: Generated new trial 4 with parameters {'x1': 0.041875, 'x2': 0.992783, 'x3': 0.906027, 'x4': 0.594429, 'x5': 0.825393, 'x6': 0.646362, 'iterations': 100}.
    
    
    
    Result for objective_2faee7c0:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: 3699644e85ac439cb7c1a36ed0976307
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.530347488145437
      landscape: -0.011983676977099367
      node_ip: 127.0.0.1
      pid: 44728
      time_since_restore: 0.00022292137145996094
      time_this_iter_s: 0.00022292137145996094
      time_total_s: 0.00022292137145996094
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2faee7c0
      warmup_time: 0.0027179718017578125
      
    Result for objective_2fa776c0:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1634068454629019
      landscape: -0.11285961764770336
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 0.000225067138671875
      time_this_iter_s: 0.000225067138671875
      time_total_s: 0.000225067138671875
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2fa776c0
      warmup_time: 0.0026290416717529297
      
    Result for objective_2dfbe86a:
      date: 2022-07-22_15-04-24
      done: true
      experiment_id: 4ef8a12ac94a4f4fa483ec18e347967f
      experiment_tag: 1_iterations=100,x1=0.0558,x2=0.0896,x3=0.9590,x4=0.2345,x5=0.1745,x6=0.9703
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3991721132671366
      landscape: -0.8052333562869153
      node_ip: 127.0.0.1
      pid: 44721
      time_since_restore: 2.573719024658203
      time_this_iter_s: 0.0251619815826416
      time_total_s: 2.573719024658203
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2dfbe86a
      warmup_time: 0.0035619735717773438
      
    Result for objective_2fabaa1a:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: eb9287e4fe5f44c7868dc943e2642312
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3599537840291782
      landscape: -0.11348012497414121
      node_ip: 127.0.0.1
      pid: 44727
      time_since_restore: 0.00022077560424804688
      time_this_iter_s: 0.00022077560424804688
      time_total_s: 0.00022077560424804688
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2fabaa1a
      warmup_time: 0.0025510787963867188
      
    
    
    
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 3 with data: {'landscape': (-0.011984, None), 'l2norm': (1.530347, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 5 with parameters {'x1': 0.126064, 'x2': 0.703408, 'x3': 0.344681, 'x4': 0.337363, 'x5': 0.401396, 'x6': 0.679202, 'iterations': 100}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 1 with data: {'landscape': (-0.11286, None), 'l2norm': (1.163407, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 6 with parameters {'x1': 0.091094, 'x2': 0.304138, 'x3': 0.869848, 'x4': 0.405435, 'x5': 0.567922, 'x6': 0.228608, 'iterations': 100}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 2 with data: {'landscape': (-0.11348, None), 'l2norm': (1.359954, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 7 with parameters {'x1': 0.603178, 'x2': 0.409057, 'x3': 0.729056, 'x4': 0.082598, 'x5': 0.572948, 'x6': 0.508304, 'iterations': 100}.
    
    
    
    Result for objective_313d3d3a:
      date: 2022-07-22_15-04-27
      done: false
      experiment_id: fa7afd557e154fbebe4f54d8eedb3573
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.805729990121368
      landscape: -0.006779757704679272
      node_ip: 127.0.0.1
      pid: 44747
      time_since_restore: 0.00021076202392578125
      time_this_iter_s: 0.00021076202392578125
      time_total_s: 0.00021076202392578125
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 313d3d3a
      warmup_time: 0.0029790401458740234
      
    Result for objective_2faee7c0:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: 3699644e85ac439cb7c1a36ed0976307
      experiment_tag: 4_iterations=100,x1=0.6647,x2=0.2075,x3=0.3595,x4=0.7046,x5=0.7559,x6=0.8124
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.530347488145437
      landscape: -0.011983676977099367
      node_ip: 127.0.0.1
      pid: 44728
      time_since_restore: 2.6206929683685303
      time_this_iter_s: 0.027359962463378906
      time_total_s: 2.6206929683685303
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2faee7c0
      warmup_time: 0.0027179718017578125
      
    Result for objective_2fa776c0:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      experiment_tag: 2_iterations=100,x1=0.7448,x2=0.7545,x3=0.0950,x4=0.2739,x5=0.0967,x6=0.3689
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1634068454629019
      landscape: -0.11285961764770336
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 2.6361019611358643
      time_this_iter_s: 0.0264589786529541
      time_total_s: 2.6361019611358643
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2fa776c0
      warmup_time: 0.0026290416717529297
      
    Result for objective_32c9acd8:
      date: 2022-07-22_15-04-27
      done: false
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1686440476629836
      landscape: -0.9046216637367911
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 0.00020194053649902344
      time_this_iter_s: 0.00020194053649902344
      time_total_s: 0.00020194053649902344
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32c9acd8
      warmup_time: 0.0026290416717529297
      
    Result for objective_2fabaa1a:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: eb9287e4fe5f44c7868dc943e2642312
      experiment_tag: 3_iterations=100,x1=0.4057,x2=0.3746,x3=0.9356,x4=0.2222,x5=0.7872,x6=0.0081
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3599537840291782
      landscape: -0.11348012497414121
      node_ip: 127.0.0.1
      pid: 44727
      time_since_restore: 2.623929977416992
      time_this_iter_s: 0.032716989517211914
      time_total_s: 2.623929977416992
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2fabaa1a
      warmup_time: 0.0025510787963867188
      
    Result for objective_32d8dd20:
      date: 2022-07-22_15-04-30
      done: false
      experiment_id: 171527593b0f4cbf941c0a03faaf0953
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.2869105702896437
      landscape: -0.24722262157458608
      node_ip: 127.0.0.1
      pid: 44758
      time_since_restore: 0.00021886825561523438
      time_this_iter_s: 0.00021886825561523438
      time_total_s: 0.00021886825561523438
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32d8dd20
      warmup_time: 0.002732992172241211
      
    Result for objective_32cf8ca2:
      date: 2022-07-22_15-04-29
      done: false
      experiment_id: 37610500f6df493aae4e7e46bb21bf09
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1817810425508524
      landscape: -0.14653248187442922
      node_ip: 127.0.0.1
      pid: 44756
      time_since_restore: 0.00025081634521484375
      time_this_iter_s: 0.00025081634521484375
      time_total_s: 0.00025081634521484375
      timestamp: 1658498669
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32cf8ca2
      warmup_time: 0.0032138824462890625
      
    
    
    
    [INFO 07-22 15:04:30] ax.service.ax_client: Completed trial 4 with data: {'landscape': (-0.00678, None), 'l2norm': (1.80573, None)}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Generated new trial 8 with parameters {'x1': 0.454189, 'x2': 0.271772, 'x3': 0.530871, 'x4': 0.991841, 'x5': 0.691843, 'x6': 0.472366, 'iterations': 100}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Completed trial 5 with data: {'landscape': (-0.904622, None), 'l2norm': (1.168644, None)}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Generated new trial 9 with parameters {'x1': 0.265264, 'x2': 0.924884, 'x3': 0.151716, 'x4': 0.436026, 'x5': 0.85731, 'x6': 0.08981, 'iterations': 100}.
    
    
    
    Result for objective_313d3d3a:
      date: 2022-07-22_15-04-30
      done: true
      experiment_id: fa7afd557e154fbebe4f54d8eedb3573
      experiment_tag: 5_iterations=100,x1=0.0419,x2=0.9928,x3=0.9060,x4=0.5944,x5=0.8254,x6=0.6464
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.805729990121368
      landscape: -0.006779757704679272
      node_ip: 127.0.0.1
      pid: 44747
      time_since_restore: 3.1623308658599854
      time_this_iter_s: 0.02911996841430664
      time_total_s: 3.1623308658599854
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 313d3d3a
      warmup_time: 0.0029790401458740234
      
    Result for objective_32c9acd8:
      date: 2022-07-22_15-04-30
      done: true
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      experiment_tag: 6_iterations=100,x1=0.1261,x2=0.7034,x3=0.3447,x4=0.3374,x5=0.4014,x6=0.6792
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1686440476629836
      landscape: -0.9046216637367911
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 3.1211891174316406
      time_this_iter_s: 0.02954697608947754
      time_total_s: 3.1211891174316406
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32c9acd8
      warmup_time: 0.0026290416717529297
      
    
    
    
    [INFO 07-22 15:04:32] ax.service.ax_client: Completed trial 7 with data: {'landscape': (-0.247223, None), 'l2norm': (1.286911, None)}.
    [INFO 07-22 15:04:32] ax.service.ax_client: Completed trial 6 with data: {'landscape': (-0.146532, None), 'l2norm': (1.181781, None)}.
    
    
    
    Result for objective_32d8dd20:
      date: 2022-07-22_15-04-32
      done: true
      experiment_id: 171527593b0f4cbf941c0a03faaf0953
      experiment_tag: 8_iterations=100,x1=0.6032,x2=0.4091,x3=0.7291,x4=0.0826,x5=0.5729,x6=0.5083
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.2869105702896437
      landscape: -0.24722262157458608
      node_ip: 127.0.0.1
      pid: 44758
      time_since_restore: 2.6415798664093018
      time_this_iter_s: 0.026781082153320312
      time_total_s: 2.6415798664093018
      timestamp: 1658498672
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32d8dd20
      warmup_time: 0.002732992172241211
      
    Result for objective_32cf8ca2:
      date: 2022-07-22_15-04-32
      done: true
      experiment_id: 37610500f6df493aae4e7e46bb21bf09
      experiment_tag: 7_iterations=100,x1=0.0911,x2=0.3041,x3=0.8698,x4=0.4054,x5=0.5679,x6=0.2286
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1817810425508524
      landscape: -0.14653248187442922
      node_ip: 127.0.0.1
      pid: 44756
      time_since_restore: 2.707913875579834
      time_this_iter_s: 0.027456998825073242
      time_total_s: 2.707913875579834
      timestamp: 1658498672
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32cf8ca2
      warmup_time: 0.0032138824462890625
      
    Result for objective_34adf04a:
      date: 2022-07-22_15-04-33
      done: false
      experiment_id: 4f65c5b68f5c49d98fda388e37c83deb
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.4991655675380078
      landscape: -0.01329150870283869
      node_ip: 127.0.0.1
      pid: 44768
      time_since_restore: 0.00021600723266601562
      time_this_iter_s: 0.00021600723266601562
      time_total_s: 0.00021600723266601562
      timestamp: 1658498673
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 34adf04a
      warmup_time: 0.0027239322662353516
      
    Result for objective_34b7abda:
      date: 2022-07-22_15-04-33
      done: false
      experiment_id: f135a2c40f5644ba9d2ae096a9dd10e0
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3718451333547932
      landscape: -1.6624439263544026
      node_ip: 127.0.0.1
      pid: 44771
      time_since_restore: 0.0002338886260986328
      time_this_iter_s: 0.0002338886260986328
      time_total_s: 0.0002338886260986328
      timestamp: 1658498673
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 34b7abda
      warmup_time: 0.002721071243286133
      
    
    
    
    [INFO 07-22 15:04:35] ax.service.ax_client: Completed trial 8 with data: {'landscape': (-0.013292, None), 'l2norm': (1.499166, None)}.
    [INFO 07-22 15:04:35] ax.service.ax_client: Completed trial 9 with data: {'landscape': (-1.662444, None), 'l2norm': (1.371845, None)}.
    
    
    
    Result for objective_34adf04a:
      date: 2022-07-22_15-04-35
      done: true
      experiment_id: 4f65c5b68f5c49d98fda388e37c83deb
      experiment_tag: 9_iterations=100,x1=0.4542,x2=0.2718,x3=0.5309,x4=0.9918,x5=0.6918,x6=0.4724
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.4991655675380078
      landscape: -0.01329150870283869
      node_ip: 127.0.0.1
      pid: 44768
      time_since_restore: 2.7032668590545654
      time_this_iter_s: 0.029300928115844727
      time_total_s: 2.7032668590545654
      timestamp: 1658498675
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 34adf04a
      warmup_time: 0.0027239322662353516
      
    Result for objective_34b7abda:
      date: 2022-07-22_15-04-35
      done: true
      experiment_id: f135a2c40f5644ba9d2ae096a9dd10e0
      experiment_tag: 10_iterations=100,x1=0.2653,x2=0.9249,x3=0.1517,x4=0.4360,x5=0.8573,x6=0.0898
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3718451333547932
      landscape: -1.6624439263544026
      node_ip: 127.0.0.1
      pid: 44771
      time_since_restore: 2.6852078437805176
      time_this_iter_s: 0.029579877853393555
      time_total_s: 2.6852078437805176
      timestamp: 1658498675
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 34b7abda
      warmup_time: 0.002721071243286133
      
    

And now we have the hyperparameters found to minimize the mean loss.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'iterations': 100, 'x1': 0.26526361983269453, 'x2': 0.9248840995132923, 'x3': 0.15171580761671066, 'x4': 0.43602637108415365, 'x5': 0.8573104059323668, 'x6': 0.08981018699705601}
    

# Running Tune experiments with Optuna

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=ray-tune-optuna_example)   

In this tutorial we introduce Optuna, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Optuna and, as a result, allow you to seamlessly scale up a Optuna optimization process - without sacrificing performance.

Similar to Ray Tune, Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative (“how” over “what” emphasis), define-by-run style user API. With Optuna, a user has the ability to dynamically construct the search spaces for the hyperparameters. Optuna falls in the domain of “derivative-free optimization” and “black-box optimization”.

In this example we minimize a simple objective to briefly demonstrate the usage of Optuna with Ray Tune via `OptunaSearch`, including examples of conditional search spaces (string together relationships between hyperparameters), and the multi-objective problem (measure trade-offs among all important metrics). It’s useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `optuna>=3.0.0` library is installed. To learn more, please refer to [Optuna website](https://optuna.org/).

Please note that sophisticated schedulers, such as `AsyncHyperBandScheduler`, may not work correctly with multi-objective optimization, since they typically expect a scalar score to compare fitness among trials.

## Prerequisites
    
    
    # !pip install "ray[tune]"
    !pip install -q "optuna>=3.0.0"
    

Next, import the necessary libraries:
    
    
    import time
    from typing import Dict, Optional, Any
    
    import ray
    from ray import tune
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    
    
    
    ray.init(configure_logging=False)  # initialize Ray
    

Show code cell output Hide code cell output

Let’s start by defining a simple evaluation function. An explicit math formula is queried here for demonstration, yet in practice this is typically a black-box function– e.g. the performance results after training an ML model. We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment. This setup assumes that we’re running multiple `step`s of an experiment while tuning three hyperparameters, namely `width`, `height`, and `activation`.
    
    
    def evaluate(step, width, height, activation):
        time.sleep(0.1)
        activation_boost = 10 if activation=="relu" else 0
        return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost
    

Next, our `objective` function to be optimized takes a Tune `config`, evaluates the `score` of your experiment in a training loop, and uses `tune.report` to report the `score` back to Tune.
    
    
    def objective(config):
        for step in range(config["steps"]):
            score = evaluate(step, config["width"], config["height"], config["activation"])
            tune.report({"iterations": step, "mean_loss": score})
    

Next we define a search space. The critical assumption is that the optimal hyperparameters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.

The simplest case is a search space with independent dimensions. In this case, a config dictionary will suffice.
    
    
    search_space = {
        "steps": 100,
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
        "activation": tune.choice(["relu", "tanh"]),
    }
    

Here we define the Optuna search algorithm:
    
    
    algo = OptunaSearch()
    

We also constrain the number of concurrent trials to `4` with a `ConcurrencyLimiter`.
    
    
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    

The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples. (you can decrease this if it takes too long on your machine).
    
    
    num_samples = 1000
    

Finally, we run the experiment to `"min"`imize the “mean_loss” of the `objective` by searching `search_space` via `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    

Show code cell output Hide code cell output

### Tune Status

Current time:| 2025-02-10 18:06:12  
---|---  
Running for: | 00:00:35.68   
Memory: | 22.7/36.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs 

### Trial Status

Trial name | status | loc | activation |  height|  width|  loss|  iter|  total time (s)|  iterations  
---|---|---|---|---|---|---|---|---|---  
objective_989a402c| TERMINATED| 127.0.0.1:42307| relu |  6.57558|  8.66313| 10.7728 |  100|  10.3642|  99  
objective_d99d28c6| TERMINATED| 127.0.0.1:42321| tanh |  51.2103 | 19.2804 |  5.17314|  100|  10.3775|  99  
objective_ce34b92b| TERMINATED| 127.0.0.1:42323| tanh | -49.4554 | 17.2683 | -4.88739|  100|  10.3741|  99  
objective_f650ea5f| TERMINATED| 127.0.0.1:42332| tanh |  20.6147 |  3.19539|  2.3679 |  100|  10.3804|  99  
objective_e72e976e| TERMINATED| 127.0.0.1:42356| relu | -12.5302 |  3.45152|  9.03132|  100|  10.372 |  99  
objective_d00b4e1a| TERMINATED| 127.0.0.1:42362| tanh |  65.8592 |  3.14335|  6.89726|  100|  10.3776|  99  
objective_30c6ec86| TERMINATED| 127.0.0.1:42367| tanh | -82.0713 | 14.2595 | -8.13679|  100|  10.3755|  99  
objective_691ce63c| TERMINATED| 127.0.0.1:42368| tanh |  29.406 |  2.21881|  3.37602|  100|  10.3653|  99  
objective_3051162c| TERMINATED| 127.0.0.1:42404| relu |  61.1787 | 12.9673 | 16.1952 |  100|  10.3885|  99  
objective_04a38992| TERMINATED| 127.0.0.1:42405| relu |  6.28688| 11.4537 | 10.7161 |  100|  10.4051|  99  
  
And now we have the hyperparameters found to minimize the mean loss.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'steps': 100, 'width': 14.259467682064852, 'height': -82.07132174642958, 'activation': 'tanh'}
    

## Providing an initial set of hyperparameters

While defining the search algorithm, we may choose to provide an initial set of hyperparameters that we believe are especially promising or informative, and pass this information as a helpful starting point for the `OptunaSearch` object.
    
    
    initial_params = [
        {"width": 1, "height": 2, "activation": "relu"},
        {"width": 4, "height": 2, "activation": "relu"},
    ]
    

Now the `search_alg` built using `OptunaSearch` takes `points_to_evaluate`.
    
    
    searcher = OptunaSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(searcher, max_concurrent=4)
    

And run the experiment with initial hyperparameter evaluations:
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    

Show code cell output Hide code cell output

### Tune Status

Current time:| 2025-02-10 18:06:47  
---|---  
Running for: | 00:00:35.44   
Memory: | 22.7/36.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs 

### Trial Status

Trial name | status | loc | activation |  height|  width|  loss|  iter|  total time (s)|  iterations  
---|---|---|---|---|---|---|---|---|---  
objective_1d2e715f| TERMINATED| 127.0.0.1:42435| relu |  2 |  1 | 11.1174 |  100|  10.3556|  99  
objective_f7c2aed0| TERMINATED| 127.0.0.1:42436| relu |  2 |  4 | 10.4463 |  100|  10.3702|  99  
objective_09dcce33| TERMINATED| 127.0.0.1:42438| tanh |  28.5547 | 17.4195 |  2.91312 |  100|  10.3483|  99  
objective_b9955517| TERMINATED| 127.0.0.1:42443| tanh | -73.0995 | 13.8859 | -7.23773 |  100|  10.3682|  99  
objective_d81ebd5c| TERMINATED| 127.0.0.1:42464| relu |  -1.86597|  1.46093| 10.4601 |  100|  10.3969|  99  
objective_3f0030e7| TERMINATED| 127.0.0.1:42465| relu |  38.7166 |  1.3696 | 14.5585 |  100|  10.3741|  99  
objective_86bf6402| TERMINATED| 127.0.0.1:42470| tanh |  40.269 |  5.13015|  4.21999 |  100|  10.3769|  99  
objective_75d06a83| TERMINATED| 127.0.0.1:42471| tanh | -11.2824 |  3.10251| -0.812933|  100|  10.3695|  99  
objective_0d197811| TERMINATED| 127.0.0.1:42496| tanh |  91.7076 | 15.1032 |  9.2372 |  100|  10.3631|  99  
objective_5156451f| TERMINATED| 127.0.0.1:42497| tanh |  58.9282 |  3.96315|  6.14136 |  100|  10.4732|  99  
  
We take another look at the optimal hyperparameters.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'steps': 100, 'width': 13.885889617119432, 'height': -73.09947583621019, 'activation': 'tanh'}
    

## Conditional search spaces

Sometimes we may want to build a more complicated search space that has conditional dependencies on other hyperparameters. In this case, we pass a define-by-run function to the `search_alg` argument in `ray.tune()`.
    
    
    def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
        """Define-by-run function to construct a conditional search space.
    
        Ensure no actual computation takes place here. That should go into
        the trainable passed to ``Tuner()`` (in this example, that's
        ``objective``).
    
        For more information, see https://optuna.readthedocs.io/en/stable\
        /tutorial/10_key_features/002_configurations.html
    
        Args:
            trial: Optuna Trial object
            
        Returns:
            Dict containing constant parameters or None
        """
    
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    
        # Define-by-run allows for conditional search spaces.
        if activation == "relu":
            trial.suggest_float("width", 0, 20)
            trial.suggest_float("height", -100, 100)
        else:
            trial.suggest_float("width", -1, 21)
            trial.suggest_float("height", -101, 101)
            
        # Return all constants in a dictionary.
        return {"steps": 100}
    

As before, we create the `search_alg` from `OptunaSearch` and `ConcurrencyLimiter`, this time we define the scope of search via the `space` argument and provide no initialization. We also must specific metric and mode when using `space`.
    
    
    searcher = OptunaSearch(space=define_by_run_func, metric="mean_loss", mode="min")
    algo = ConcurrencyLimiter(searcher, max_concurrent=4)
    
    
    
    [I 2025-02-10 18:06:47,670] A new study created in memory with name: optuna
    

Running the experiment with a define-by-run search space:
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    

Show code cell output Hide code cell output

### Tune Status

Current time:| 2025-02-10 18:07:23  
---|---  
Running for: | 00:00:35.58   
Memory: | 22.9/36.0 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs 

### Trial Status

Trial name | status | loc | activation |  height|  steps|  width|  loss|  iter|  total time (s)|  iterations  
---|---|---|---|---|---|---|---|---|---|---  
objective_48aa8fed| TERMINATED| 127.0.0.1:42529| relu | -76.595 |  100|  9.90896 |  2.44141|  100|  10.3957|  99  
objective_5f395194| TERMINATED| 127.0.0.1:42531| relu | -34.1447 |  100| 12.9999 |  6.66263|  100|  10.3823|  99  
objective_e64a7441| TERMINATED| 127.0.0.1:42532| relu | -50.3172 |  100|  3.95399 |  5.21738|  100|  10.3839|  99  
objective_8e668790| TERMINATED| 127.0.0.1:42537| tanh |  30.9768 |  100| 16.22 |  3.15957|  100|  10.3818|  99  
objective_78ca576b| TERMINATED| 127.0.0.1:42559| relu |  80.5037 |  100|  0.906139| 19.0533 |  100|  10.3731|  99  
objective_4cd9e37a| TERMINATED| 127.0.0.1:42560| relu |  77.0988 |  100|  8.43807 | 17.8282 |  100|  10.3881|  99  
objective_a40498d5
    
    
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
    

The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples. You can decrease this if it takes too long on your machine, or you can set a time limit easily through `stop` argument in the `RunConfig()` as we will show here.
    
    
    num_samples = 100
    stop_timesteps = 200
    

Finally, we run the experiment to find the global minimum of the provided landscape (which contains 5 false minima). The argument to metric, `"landscape"`, is provided via the `objective` function’s `tune.report`. The experiment `"min"`imizes the “mean_loss” of the `landscape` by searching within `search_space` via `algo`, `num_samples` times or when `"timesteps_total": stop_timesteps`. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.
    
    
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="landscape",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        run_config=tune.RunConfig(
            name="ax",
            stop={"timesteps_total": stop_timesteps}
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    
    
    [INFO 07-22 15:04:18] ax.service.ax_client: Starting optimization with verbose logging. To disable logging, set the `verbose_logging` argument to `False`. Note that float values in the logs are rounded to 6 decimal points.
    [INFO 07-22 15:04:18] ax.service.utils.instantiation: Created search space: SearchSpace(parameters=[FixedParameter(name='iterations', parameter_type=INT, value=100), RangeParameter(name='x1', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x2', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x3', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x4', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x5', parameter_type=FLOAT, range=[0.0, 1.0]), RangeParameter(name='x6', parameter_type=FLOAT, range=[0.0, 1.0])], parameter_constraints=[ParameterConstraint(1.0*x1 + 1.0*x2 <= 2.0)]).
    [INFO 07-22 15:04:18] ax.modelbridge.dispatch_utils: Using Bayesian optimization since there are more ordered parameters than there are categories for the unordered categorical parameters.
    [INFO 07-22 15:04:18] ax.modelbridge.dispatch_utils: Using Bayesian Optimization generation strategy: GenerationStrategy(name='Sobol+GPEI', steps=[Sobol for 12 trials, GPEI for subsequent trials]). Iterations after 12 will take longer to generate due to  model-fitting.
    Detected sequential enforcement. Be sure to use a ConcurrencyLimiter.
    

== Status ==  
Current time: 2022-07-22 15:04:35 (running for 00:00:16.56)  
Memory usage on this node: 9.9/16.0 GiB  
Using FIFO scheduling algorithm.  
Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/5.13 GiB heap, 0.0/2.0 GiB objects  
Current best trial: 34b7abda with landscape=-1.6624439263544026 and parameters={'iterations': 100, 'x1': 0.26526361983269453, 'x2': 0.9248840995132923, 'x3': 0.15171580761671066, 'x4': 0.43602637108415365, 'x5': 0.8573104059323668, 'x6': 0.08981018699705601}  
Result logdir: /Users/kai/ray_results/ax  
Number of trials: 10/10 (10 TERMINATED)  
| Trial name | status | loc |  iterations|  x1|  x2|  x3|  x4|  x5|  x6|  iter|  total time (s)|  ts|  landscape|  l2norm  
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---  
objective_2dfbe86a| TERMINATED| 127.0.0.1:44721|  100| 0.0558336| 0.0896192| 0.958956 | 0.234474 | 0.174516 | 0.970311 |  100|  2.57372|  99| -0.805233 |  1.39917  
objective_2fa776c0| TERMINATED| 127.0.0.1:44726|  100| 0.744772 | 0.754537 | 0.0950125| 0.273877 | 0.0966829| 0.368943 |  100|  2.6361 |  99| -0.11286 |  1.16341  
objective_2fabaa1a| TERMINATED| 127.0.0.1:44727|  100| 0.405704 | 0.374626 | 0.935628 | 0.222185 | 0.787212 | 0.00812439|  100|  2.62393|  99| -0.11348 |  1.35995  
objective_2faee7c0| TERMINATED| 127.0.0.1:44728|  100| 0.664728 | 0.207519 | 0.359514 | 0.704578 | 0.755882 | 0.812402 |  100|  2.62069|  99| -0.0119837 |  1.53035  
objective_313d3d3a| TERMINATED| 127.0.0.1:44747|  100| 0.0418746| 0.992783 | 0.906027 | 0.594429 | 0.825393 | 0.646362 |  100|  3.16233|  99| -0.00677976|  1.80573  
objective_32c9acd8| TERMINATED| 127.0.0.1:44726|  100| 0.126064 | 0.703408 | 0.344681 | 0.337363 | 0.401396 | 0.679202 |  100|  3.12119|  99| -0.904622 |  1.16864  
objective_32cf8ca2| TERMINATED| 127.0.0.1:44756|  100| 0.0910936| 0.304138 | 0.869848 | 0.405435 | 0.567922 | 0.228608 |  100|  2.70791|  99| -0.146532 |  1.18178  
objective_32d8dd20| TERMINATED| 127.0.0.1:44758|  100| 0.603178 | 0.409057 | 0.729056 | 0.0825984| 0.572948 | 0.508304 |  100|  2.64158|  99| -0.247223 |  1.28691  
objective_34adf04a| TERMINATED| 127.0.0.1:44768|  100| 0.454189 | 0.271772 | 0.530871 | 0.991841 | 0.691843 | 0.472366 |  100|  2.70327|  99| -0.0132915 |  1.49917  
objective_34b7abda| TERMINATED| 127.0.0.1:44771|  100| 0.265264 | 0.924884 | 0.151716 | 0.436026 | 0.85731 | 0.0898102 |  100|  2.68521|  99| -1.66244 |  1.37185  
  
  

    
    
    [INFO 07-22 15:04:19] ax.service.ax_client: Generated new trial 0 with parameters {'x1': 0.055834, 'x2': 0.089619, 'x3': 0.958956, 'x4': 0.234474, 'x5': 0.174516, 'x6': 0.970311, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 1 with parameters {'x1': 0.744772, 'x2': 0.754537, 'x3': 0.095012, 'x4': 0.273877, 'x5': 0.096683, 'x6': 0.368943, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 2 with parameters {'x1': 0.405704, 'x2': 0.374626, 'x3': 0.935628, 'x4': 0.222185, 'x5': 0.787212, 'x6': 0.008124, 'iterations': 100}.
    [INFO 07-22 15:04:22] ax.service.ax_client: Generated new trial 3 with parameters {'x1': 0.664728, 'x2': 0.207519, 'x3': 0.359514, 'x4': 0.704578, 'x5': 0.755882, 'x6': 0.812402, 'iterations': 100}.
    
    
    
    Result for objective_2dfbe86a:
      date: 2022-07-22_15-04-22
      done: false
      experiment_id: 4ef8a12ac94a4f4fa483ec18e347967f
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3991721132671366
      landscape: -0.8052333562869153
      node_ip: 127.0.0.1
      pid: 44721
      time_since_restore: 0.00022912025451660156
      time_this_iter_s: 0.00022912025451660156
      time_total_s: 0.00022912025451660156
      timestamp: 1658498662
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2dfbe86a
      warmup_time: 0.0035619735717773438
      
    
    
    
    [INFO 07-22 15:04:24] ax.service.ax_client: Completed trial 0 with data: {'landscape': (-0.805233, None), 'l2norm': (1.399172, None)}.
    [INFO 07-22 15:04:24] ax.service.ax_client: Generated new trial 4 with parameters {'x1': 0.041875, 'x2': 0.992783, 'x3': 0.906027, 'x4': 0.594429, 'x5': 0.825393, 'x6': 0.646362, 'iterations': 100}.
    
    
    
    Result for objective_2faee7c0:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: 3699644e85ac439cb7c1a36ed0976307
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.530347488145437
      landscape: -0.011983676977099367
      node_ip: 127.0.0.1
      pid: 44728
      time_since_restore: 0.00022292137145996094
      time_this_iter_s: 0.00022292137145996094
      time_total_s: 0.00022292137145996094
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2faee7c0
      warmup_time: 0.0027179718017578125
      
    Result for objective_2fa776c0:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1634068454629019
      landscape: -0.11285961764770336
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 0.000225067138671875
      time_this_iter_s: 0.000225067138671875
      time_total_s: 0.000225067138671875
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2fa776c0
      warmup_time: 0.0026290416717529297
      
    Result for objective_2dfbe86a:
      date: 2022-07-22_15-04-24
      done: true
      experiment_id: 4ef8a12ac94a4f4fa483ec18e347967f
      experiment_tag: 1_iterations=100,x1=0.0558,x2=0.0896,x3=0.9590,x4=0.2345,x5=0.1745,x6=0.9703
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3991721132671366
      landscape: -0.8052333562869153
      node_ip: 127.0.0.1
      pid: 44721
      time_since_restore: 2.573719024658203
      time_this_iter_s: 0.0251619815826416
      time_total_s: 2.573719024658203
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2dfbe86a
      warmup_time: 0.0035619735717773438
      
    Result for objective_2fabaa1a:
      date: 2022-07-22_15-04-24
      done: false
      experiment_id: eb9287e4fe5f44c7868dc943e2642312
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3599537840291782
      landscape: -0.11348012497414121
      node_ip: 127.0.0.1
      pid: 44727
      time_since_restore: 0.00022077560424804688
      time_this_iter_s: 0.00022077560424804688
      time_total_s: 0.00022077560424804688
      timestamp: 1658498664
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 2fabaa1a
      warmup_time: 0.0025510787963867188
      
    
    
    
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 3 with data: {'landscape': (-0.011984, None), 'l2norm': (1.530347, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 5 with parameters {'x1': 0.126064, 'x2': 0.703408, 'x3': 0.344681, 'x4': 0.337363, 'x5': 0.401396, 'x6': 0.679202, 'iterations': 100}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 1 with data: {'landscape': (-0.11286, None), 'l2norm': (1.163407, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 6 with parameters {'x1': 0.091094, 'x2': 0.304138, 'x3': 0.869848, 'x4': 0.405435, 'x5': 0.567922, 'x6': 0.228608, 'iterations': 100}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Completed trial 2 with data: {'landscape': (-0.11348, None), 'l2norm': (1.359954, None)}.
    [INFO 07-22 15:04:27] ax.service.ax_client: Generated new trial 7 with parameters {'x1': 0.603178, 'x2': 0.409057, 'x3': 0.729056, 'x4': 0.082598, 'x5': 0.572948, 'x6': 0.508304, 'iterations': 100}.
    
    
    
    Result for objective_313d3d3a:
      date: 2022-07-22_15-04-27
      done: false
      experiment_id: fa7afd557e154fbebe4f54d8eedb3573
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.805729990121368
      landscape: -0.006779757704679272
      node_ip: 127.0.0.1
      pid: 44747
      time_since_restore: 0.00021076202392578125
      time_this_iter_s: 0.00021076202392578125
      time_total_s: 0.00021076202392578125
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 313d3d3a
      warmup_time: 0.0029790401458740234
      
    Result for objective_2faee7c0:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: 3699644e85ac439cb7c1a36ed0976307
      experiment_tag: 4_iterations=100,x1=0.6647,x2=0.2075,x3=0.3595,x4=0.7046,x5=0.7559,x6=0.8124
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.530347488145437
      landscape: -0.011983676977099367
      node_ip: 127.0.0.1
      pid: 44728
      time_since_restore: 2.6206929683685303
      time_this_iter_s: 0.027359962463378906
      time_total_s: 2.6206929683685303
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2faee7c0
      warmup_time: 0.0027179718017578125
      
    Result for objective_2fa776c0:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      experiment_tag: 2_iterations=100,x1=0.7448,x2=0.7545,x3=0.0950,x4=0.2739,x5=0.0967,x6=0.3689
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1634068454629019
      landscape: -0.11285961764770336
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 2.6361019611358643
      time_this_iter_s: 0.0264589786529541
      time_total_s: 2.6361019611358643
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2fa776c0
      warmup_time: 0.0026290416717529297
      
    Result for objective_32c9acd8:
      date: 2022-07-22_15-04-27
      done: false
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1686440476629836
      landscape: -0.9046216637367911
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 0.00020194053649902344
      time_this_iter_s: 0.00020194053649902344
      time_total_s: 0.00020194053649902344
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32c9acd8
      warmup_time: 0.0026290416717529297
      
    Result for objective_2fabaa1a:
      date: 2022-07-22_15-04-27
      done: true
      experiment_id: eb9287e4fe5f44c7868dc943e2642312
      experiment_tag: 3_iterations=100,x1=0.4057,x2=0.3746,x3=0.9356,x4=0.2222,x5=0.7872,x6=0.0081
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3599537840291782
      landscape: -0.11348012497414121
      node_ip: 127.0.0.1
      pid: 44727
      time_since_restore: 2.623929977416992
      time_this_iter_s: 0.032716989517211914
      time_total_s: 2.623929977416992
      timestamp: 1658498667
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 2fabaa1a
      warmup_time: 0.0025510787963867188
      
    Result for objective_32d8dd20:
      date: 2022-07-22_15-04-30
      done: false
      experiment_id: 171527593b0f4cbf941c0a03faaf0953
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.2869105702896437
      landscape: -0.24722262157458608
      node_ip: 127.0.0.1
      pid: 44758
      time_since_restore: 0.00021886825561523438
      time_this_iter_s: 0.00021886825561523438
      time_total_s: 0.00021886825561523438
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32d8dd20
      warmup_time: 0.002732992172241211
      
    Result for objective_32cf8ca2:
      date: 2022-07-22_15-04-29
      done: false
      experiment_id: 37610500f6df493aae4e7e46bb21bf09
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.1817810425508524
      landscape: -0.14653248187442922
      node_ip: 127.0.0.1
      pid: 44756
      time_since_restore: 0.00025081634521484375
      time_this_iter_s: 0.00025081634521484375
      time_total_s: 0.00025081634521484375
      timestamp: 1658498669
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 32cf8ca2
      warmup_time: 0.0032138824462890625
      
    
    
    
    [INFO 07-22 15:04:30] ax.service.ax_client: Completed trial 4 with data: {'landscape': (-0.00678, None), 'l2norm': (1.80573, None)}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Generated new trial 8 with parameters {'x1': 0.454189, 'x2': 0.271772, 'x3': 0.530871, 'x4': 0.991841, 'x5': 0.691843, 'x6': 0.472366, 'iterations': 100}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Completed trial 5 with data: {'landscape': (-0.904622, None), 'l2norm': (1.168644, None)}.
    [INFO 07-22 15:04:30] ax.service.ax_client: Generated new trial 9 with parameters {'x1': 0.265264, 'x2': 0.924884, 'x3': 0.151716, 'x4': 0.436026, 'x5': 0.85731, 'x6': 0.08981, 'iterations': 100}.
    
    
    
    Result for objective_313d3d3a:
      date: 2022-07-22_15-04-30
      done: true
      experiment_id: fa7afd557e154fbebe4f54d8eedb3573
      experiment_tag: 5_iterations=100,x1=0.0419,x2=0.9928,x3=0.9060,x4=0.5944,x5=0.8254,x6=0.6464
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.805729990121368
      landscape: -0.006779757704679272
      node_ip: 127.0.0.1
      pid: 44747
      time_since_restore: 3.1623308658599854
      time_this_iter_s: 0.02911996841430664
      time_total_s: 3.1623308658599854
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 313d3d3a
      warmup_time: 0.0029790401458740234
      
    Result for objective_32c9acd8:
      date: 2022-07-22_15-04-30
      done: true
      experiment_id: c555bfed13ac43e5b8c8e9f6d4b9b2f7
      experiment_tag: 6_iterations=100,x1=0.1261,x2=0.7034,x3=0.3447,x4=0.3374,x5=0.4014,x6=0.6792
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1686440476629836
      landscape: -0.9046216637367911
      node_ip: 127.0.0.1
      pid: 44726
      time_since_restore: 3.1211891174316406
      time_this_iter_s: 0.02954697608947754
      time_total_s: 3.1211891174316406
      timestamp: 1658498670
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32c9acd8
      warmup_time: 0.0026290416717529297
      
    
    
    
    [INFO 07-22 15:04:32] ax.service.ax_client: Completed trial 7 with data: {'landscape': (-0.247223, None), 'l2norm': (1.286911, None)}.
    [INFO 07-22 15:04:32] ax.service.ax_client: Completed trial 6 with data: {'landscape': (-0.146532, None), 'l2norm': (1.181781, None)}.
    
    
    
    Result for objective_32d8dd20:
      date: 2022-07-22_15-04-32
      done: true
      experiment_id: 171527593b0f4cbf941c0a03faaf0953
      experiment_tag: 8_iterations=100,x1=0.6032,x2=0.4091,x3=0.7291,x4=0.0826,x5=0.5729,x6=0.5083
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.2869105702896437
      landscape: -0.24722262157458608
      node_ip: 127.0.0.1
      pid: 44758
      time_since_restore: 2.6415798664093018
      time_this_iter_s: 0.026781082153320312
      time_total_s: 2.6415798664093018
      timestamp: 1658498672
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32d8dd20
      warmup_time: 0.002732992172241211
      
    Result for objective_32cf8ca2:
      date: 2022-07-22_15-04-32
      done: true
      experiment_id: 37610500f6df493aae4e7e46bb21bf09
      experiment_tag: 7_iterations=100,x1=0.0911,x2=0.3041,x3=0.8698,x4=0.4054,x5=0.5679,x6=0.2286
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.1817810425508524
      landscape: -0.14653248187442922
      node_ip: 127.0.0.1
      pid: 44756
      time_since_restore: 2.707913875579834
      time_this_iter_s: 0.027456998825073242
      time_total_s: 2.707913875579834
      timestamp: 1658498672
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 32cf8ca2
      warmup_time: 0.0032138824462890625
      
    Result for objective_34adf04a:
      date: 2022-07-22_15-04-33
      done: false
      experiment_id: 4f65c5b68f5c49d98fda388e37c83deb
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.4991655675380078
      landscape: -0.01329150870283869
      node_ip: 127.0.0.1
      pid: 44768
      time_since_restore: 0.00021600723266601562
      time_this_iter_s: 0.00021600723266601562
      time_total_s: 0.00021600723266601562
      timestamp: 1658498673
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 34adf04a
      warmup_time: 0.0027239322662353516
      
    Result for objective_34b7abda:
      date: 2022-07-22_15-04-33
      done: false
      experiment_id: f135a2c40f5644ba9d2ae096a9dd10e0
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 1
      l2norm: 1.3718451333547932
      landscape: -1.6624439263544026
      node_ip: 127.0.0.1
      pid: 44771
      time_since_restore: 0.0002338886260986328
      time_this_iter_s: 0.0002338886260986328
      time_total_s: 0.0002338886260986328
      timestamp: 1658498673
      timesteps_since_restore: 0
      timesteps_total: 0
      training_iteration: 1
      trial_id: 34b7abda
      warmup_time: 0.002721071243286133
      
    
    
    
    [INFO 07-22 15:04:35] ax.service.ax_client: Completed trial 8 with data: {'landscape': (-0.013292, None), 'l2norm': (1.499166, None)}.
    [INFO 07-22 15:04:35] ax.service.ax_client: Completed trial 9 with data: {'landscape': (-1.662444, None), 'l2norm': (1.371845, None)}.
    
    
    
    Result for objective_34adf04a:
      date: 2022-07-22_15-04-35
      done: true
      experiment_id: 4f65c5b68f5c49d98fda388e37c83deb
      experiment_tag: 9_iterations=100,x1=0.4542,x2=0.2718,x3=0.5309,x4=0.9918,x5=0.6918,x6=0.4724
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.4991655675380078
      landscape: -0.01329150870283869
      node_ip: 127.0.0.1
      pid: 44768
      time_since_restore: 2.7032668590545654
      time_this_iter_s: 0.029300928115844727
      time_total_s: 2.7032668590545654
      timestamp: 1658498675
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 34adf04a
      warmup_time: 0.0027239322662353516
      
    Result for objective_34b7abda:
      date: 2022-07-22_15-04-35
      done: true
      experiment_id: f135a2c40f5644ba9d2ae096a9dd10e0
      experiment_tag: 10_iterations=100,x1=0.2653,x2=0.9249,x3=0.1517,x4=0.4360,x5=0.8573,x6=0.0898
      hostname: Kais-MacBook-Pro.local
      iterations_since_restore: 100
      l2norm: 1.3718451333547932
      landscape: -1.6624439263544026
      node_ip: 127.0.0.1
      pid: 44771
      time_since_restore: 2.6852078437805176
      time_this_iter_s: 0.029579877853393555
      time_total_s: 2.6852078437805176
      timestamp: 1658498675
      timesteps_since_restore: 0
      timesteps_total: 99
      training_iteration: 100
      trial_id: 34b7abda
      warmup_time: 0.002721071243286133
      
    

And now we have the hyperparameters found to minimize the mean loss.
    
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    
    Best hyperparameters found were:  {'iterations': 100, 'x1': 0.26526361983269453, 'x2': 0.9248840995132923, 'x3': 0.15171580761671066, 'x4': 0.43602637108415365, 'x5': 0.8573104059323668, 'x6': 0.08981018699705601}