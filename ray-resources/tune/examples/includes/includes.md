# Tune Examples - Includes

This file consolidates all example code snippets used in Tune documentation.

## Asynchronous HyperBand Example

# Asynchronous HyperBand Example

This example demonstrates how to use Ray Tune’s Asynchronous Successive Halving Algorithm (ASHA) scheduler to efficiently optimize hyperparameters for a machine learning model. ASHA is particularly useful for large-scale hyperparameter optimization as it can adaptively allocate resources and end poorly performing trials early.

Requirements: `pip install "ray[tune]"`
    
    
    #!/usr/bin/env python
    
    import argparse
    import time
    from typing import Any, Dict
    
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler
    
    
    def evaluation_fn(step, width, height) -> float:
        # simulate model evaluation
        time.sleep(0.1)
        return (0.1 + width * step / 100) ** (-1) + height * 0.1
    
    
    def easy_objective(config: Dict[str, Any]) -> None:
        # Config contains the hyperparameters to tune
        width, height = config["width"], config["height"]
    
        for step in range(config["steps"]):
            # Iterative training function - can be an arbitrary training procedure
            intermediate_score = evaluation_fn(step, width, height)
            # Feed the score back back to Tune.
            tune.report({"iterations": step, "mean_loss": intermediate_score})
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="AsyncHyperBand optimization example")
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        # AsyncHyperBand enables aggressive early stopping of poorly performing trials
        scheduler = AsyncHyperBandScheduler(
            grace_period=5,  # Minimum training iterations before stopping
            max_t=100,  # Maximum training iterations
        )
    
        tuner = tune.Tuner(
            tune.with_resources(easy_objective, {"cpu": 1, "gpu": 0}),
            run_config=tune.RunConfig(
                name="asynchyperband_test",
                stop={"training_iteration": 1 if args.smoke_test else 9999},
                verbose=1,
            ),
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=20,  # Number of trials to run
            ),
            param_space={
                "steps": 100,
                "width": tune.uniform(10, 100),
                "height": tune.uniform(0, 100),
            },
        )
    
        # Run the hyperparameter optimization
        results = tuner.fit()
        print(f"Best hyperparameters found: {results.get_best_result().config}")
    

## See Also

  * [ASHA Paper](https://arxiv.org/abs/1810.05934)

---

## HyperBand Function Example

# HyperBand Function Example
    
    
    #!/usr/bin/env python
    
    import argparse
    import json
    import os
    import tempfile
    
    import numpy as np
    
    import ray
    from ray import tune
    from ray.tune import Checkpoint
    from ray.tune.schedulers import HyperBandScheduler
    
    
    def train_func(config):
        step = 0
        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, "checkpoint.json")) as f:
                    step = json.load(f)["timestep"] + 1
    
        for timestep in range(step, 100):
            v = np.tanh(float(timestep) / config.get("width", 1))
            v *= config.get("height", 1)
    
            # Checkpoint the state of the training every 3 steps
            # Note that this is only required for certain schedulers
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if timestep % 3 == 0:
                    with open(
                        os.path.join(temp_checkpoint_dir, "checkpoint.json"), "w"
                    ) as f:
                        json.dump({"timestep": timestep}, f)
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
    
                # Here we use `episode_reward_mean`, but you can also report other
                # objectives such as loss or accuracy.
                tune.report({"episode_reward_mean": v}, checkpoint=checkpoint)
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        ray.init(num_cpus=4 if args.smoke_test else None)
    
        # Hyperband early stopping, configured with `episode_reward_mean` as the
        # objective and `training_iteration` as the time unit,
        # which is automatically filled by Tune.
        hyperband = HyperBandScheduler(max_t=200)
    
        tuner = tune.Tuner(
            train_func,
            run_config=tune.RunConfig(
                name="hyperband_test",
                stop={"training_iteration": 10 if args.smoke_test else 99999},
                failure_config=tune.FailureConfig(
                    fail_fast=True,
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=20,
                metric="episode_reward_mean",
                mode="max",
                scheduler=hyperband,
            ),
            param_space={"height": tune.uniform(0, 100)},
        )
        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)

---

## Logging Example

# Logging Example
    
    
    #!/usr/bin/env python
    
    import argparse
    import time
    
    from ray import tune
    from ray.tune.logger import LoggerCallback
    
    
    class TestLoggerCallback(LoggerCallback):
        def on_trial_result(self, iteration, trials, trial, result, **info):
            print(f"TestLogger for trial {trial}: {result}")
    
    
    def trial_str_creator(trial):
        return "{}_{}_123".format(trial.trainable_name, trial.trial_id)
    
    
    def evaluation_fn(step, width, height):
        time.sleep(0.1)
        return (0.1 + width * step / 100) ** (-1) + height * 0.1
    
    
    def easy_objective(config):
        # Hyperparameters
        width, height = config["width"], config["height"]
    
        for step in range(config["steps"]):
            # Iterative training function - can be any arbitrary training procedure
            intermediate_score = evaluation_fn(step, width, height)
            # Feed the score back back to Tune.
            tune.report({"iterations": step, "mean_loss": intermediate_score})
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        tuner = tune.Tuner(
            easy_objective,
            run_config=tune.RunConfig(
                name="hyperband_test",
                callbacks=[TestLoggerCallback()],
                stop={"training_iteration": 1 if args.smoke_test else 100},
            ),
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                num_samples=5,
                trial_name_creator=trial_str_creator,
                trial_dirname_creator=trial_str_creator,
            ),
            param_space={
                "steps": 100,
                "width": tune.randint(10, 100),
                "height": tune.loguniform(10, 100),
            },
        )
        results = tuner.fit()
    
        print("Best hyperparameters: ", results.get_best_result().config)

---

## MLflow PyTorch Lightning Example

# MLflow PyTorch Lightning Example
    
    
    """An example showing how to use Pytorch Lightning training, Ray Tune
    HPO, and MLflow autologging all together."""
    
    import os
    import tempfile
    
    import mlflow
    import pytorch_lightning as pl
    
    from ray import tune
    from ray.air.integrations.mlflow import setup_mlflow
    from ray.tune.examples.mnist_ptl_mini import LightningMNISTClassifier, MNISTDataModule
    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    
    
    def train_mnist_tune(config, data_dir=None, num_epochs=10, num_gpus=0):
        setup_mlflow(
            config,
            experiment_name=config.get("experiment_name", None),
            tracking_uri=config.get("tracking_uri", None),
        )
    
        model = LightningMNISTClassifier(config, data_dir)
        dm = MNISTDataModule(
            data_dir=data_dir, num_workers=1, batch_size=config["batch_size"]
        )
        metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
        mlflow.pytorch.autolog()
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=num_gpus,
            progress_bar_refresh_rate=0,
            callbacks=[TuneReportCallback(metrics, on="validation_end")],
        )
        trainer.fit(model, dm)
    
    
    def tune_mnist(
        num_samples=10,
        num_epochs=10,
        gpus_per_trial=0,
        tracking_uri=None,
        experiment_name="ptl_autologging_example",
    ):
        data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
        # Download data
        MNISTDataModule(data_dir=data_dir, batch_size=32).prepare_data()
    
        # Set the MLflow experiment, or create it if it does not exist.
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
        config = {
            "layer_1": tune.choice([32, 64, 128]),
            "layer_2": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "data_dir": os.path.join(tempfile.gettempdir(), "mnist_data_"),
            "num_epochs": num_epochs,
        }
    
        trainable = tune.with_parameters(
            train_mnist_tune,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
        )
    
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources={"cpu": 1, "gpu": gpus_per_trial}),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=num_samples,
            ),
            run_config=tune.RunConfig(
                name="tune_mnist",
            ),
            param_space=config,
        )
        results = tuner.fit()
    
        print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    if __name__ == "__main__":
        import argparse
    
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        if args.smoke_test:
            tune_mnist(
                num_samples=1,
                num_epochs=1,
                gpus_per_trial=0,
                tracking_uri=os.path.join(tempfile.gettempdir(), "mlruns"),
            )
        else:
            tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=0)

---

## MNIST PyTorch Example

# MNIST PyTorch Example
    
    
    # Original Code here:
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    
    import argparse
    import os
    import tempfile
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from filelock import FileLock
    from torchvision import datasets, transforms
    
    import ray
    from ray import tune
    from ray.tune import Checkpoint
    from ray.tune.schedulers import AsyncHyperBandScheduler
    
    # Change these values if you want the training to run quicker or slower.
    EPOCH_SIZE = 512
    TEST_SIZE = 256
    
    
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
            self.fc = nn.Linear(192, 10)
    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 3))
            x = x.view(-1, 192)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)
    
    
    def train_func(model, optimizer, train_loader, device=None):
        device = device or torch.device("cpu")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * len(data) > EPOCH_SIZE:
                return
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
    
    def test_func(model, data_loader, device=None):
        device = device or torch.device("cpu")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx * len(data) > TEST_SIZE:
                    break
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
    
        return correct / total
    
    
    def get_data_loaders(batch_size=64):
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    
        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("~/data.lock")):
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    "~/data", train=True, download=True, transform=mnist_transforms
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    "~/data", train=False, download=True, transform=mnist_transforms
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        return train_loader, test_loader
    
    
    def train_mnist(config):
        should_checkpoint = config.get("should_checkpoint", False)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train_loader, test_loader = get_data_loaders()
        model = ConvNet().to(device)
    
        optimizer = optim.SGD(
            model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )
    
        while True:
            train_func(model, optimizer, train_loader, device)
            acc = test_func(model, test_loader, device)
            metrics = {"mean_accuracy": acc}
    
            # Report metrics (and possibly a checkpoint)
            if should_checkpoint:
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                    tune.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                tune.report(metrics)
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
        parser.add_argument(
            "--cuda", action="store_true", default=False, help="Enables GPU training"
        )
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        ray.init(num_cpus=2 if args.smoke_test else None)
    
        # for early stopping
        sched = AsyncHyperBandScheduler()
    
        resources_per_trial = {"cpu": 2, "gpu": int(args.cuda)}  # set this for GPUs
        tuner = tune.Tuner(
            tune.with_resources(train_mnist, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                metric="mean_accuracy",
                mode="max",
                scheduler=sched,
                num_samples=1 if args.smoke_test else 50,
            ),
            run_config=tune.RunConfig(
                name="exp",
                stop={
                    "mean_accuracy": 0.98,
                    "training_iteration": 5 if args.smoke_test else 100,
                },
            ),
            param_space={
                "lr": tune.loguniform(1e-4, 1e-2),
                "momentum": tune.uniform(0.1, 0.9),
            },
        )
        results = tuner.fit()
    
        print("Best config is:", results.get_best_result().config)
    
        assert not results.errors
    

If you consider switching to PyTorch Lightning to get rid of some of your boilerplate training code, please know that we also have a walkthrough on [how to use Tune with PyTorch Lightning models](../tune-pytorch-lightning.md).

---

## MNIST PyTorch Trainable Example

# MNIST PyTorch Trainable Example
    
    
    # Original Code here:
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    from __future__ import print_function
    
    import argparse
    import os
    
    import torch
    import torch.optim as optim
    
    import ray
    from ray import tune
    from ray.tune.examples.mnist_pytorch import (
        ConvNet,
        get_data_loaders,
        test_func,
        train_func,
    )
    from ray.tune.schedulers import ASHAScheduler
    
    # Change these values if you want the training to run quicker or slower.
    EPOCH_SIZE = 512
    TEST_SIZE = 256
    
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    
    
    # Below comments are for documentation purposes only.
    # fmt: off
    # __trainable_example_begin__
    class TrainMNIST(tune.Trainable):
        def setup(self, config):
            use_cuda = config.get("use_gpu") and torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
            self.train_loader, self.test_loader = get_data_loaders()
            self.model = ConvNet().to(self.device)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.get("lr", 0.01),
                momentum=config.get("momentum", 0.9))
    
        def step(self):
            train_func(
                self.model, self.optimizer, self.train_loader, device=self.device)
            acc = test_func(self.model, self.test_loader, self.device)
            return {"mean_accuracy": acc}
    
        def save_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
    
        def load_checkpoint(self, checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            self.model.load_state_dict(torch.load(checkpoint_path))
    
    
    # __trainable_example_end__
    # fmt: on
    
    if __name__ == "__main__":
        args = parser.parse_args()
        ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)
        sched = ASHAScheduler()
    
        tuner = tune.Tuner(
            tune.with_resources(TrainMNIST, resources={"cpu": 3, "gpu": int(args.use_gpu)}),
            run_config=tune.RunConfig(
                stop={
                    "mean_accuracy": 0.95,
                    "training_iteration": 3 if args.smoke_test else 20,
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True, checkpoint_frequency=3
                ),
            ),
            tune_config=tune.TuneConfig(
                metric="mean_accuracy",
                mode="max",
                scheduler=sched,
                num_samples=1 if args.smoke_test else 20,
            ),
            param_space={
                "args": args,
                "lr": tune.uniform(0.001, 0.1),
                "momentum": tune.uniform(0.1, 0.9),
            },
        )
        results = tuner.fit()
    
        print("Best config is:", results.get_best_result().config)

---

## PB2 Example

# PB2 Example
    
    
    #!/usr/bin/env python
    
    import argparse
    
    import ray
    from ray import tune
    from ray.tune.examples.pbt_function import pbt_function
    from ray.tune.schedulers.pb2 import PB2
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        if args.smoke_test:
            ray.init(num_cpus=2)  # force pausing to happen for test
    
        perturbation_interval = 5
        pbt = PB2(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            hyperparam_bounds={
                # hyperparameter bounds.
                "lr": [0.0001, 0.02],
            },
        )
    
        tuner = tune.Tuner(
            pbt_function,
            run_config=tune.RunConfig(
                name="pbt_test",
                verbose=False,
                stop={
                    "training_iteration": 30,
                },
                failure_config=tune.FailureConfig(
                    fail_fast=True,
                ),
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                metric="mean_accuracy",
                mode="max",
                num_samples=8,
                reuse_actors=True,
            ),
            param_space={
                "lr": 0.0001,
                # note: this parameter is perturbed but has no effect on
                # the model training in this example
                "some_other_factor": 1,
                # This parameter is not perturbed and is used to determine
                # checkpoint frequency. We set checkpoints and perturbations
                # to happen at the same frequency.
                "checkpoint_interval": perturbation_interval,
            },
        )
        results = tuner.fit()
    
        print("Best hyperparameters found were: ", results.get_best_result().config)

---

## PB2 PPO Example

# PB2 PPO Example
    
    
    import argparse
    import os
    import random
    from datetime import datetime
    
    import pandas as pd
    
    from ray.tune import run, sample_from
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune.schedulers.pb2 import PB2
    
    
    # Postprocess the perturbed config to ensure it's still valid used if PBT.
    def explore(config):
        # Ensure we collect enough timesteps to do sgd.
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # Ensure we run at least one sgd iter.
        if config["lambda"] > 1:
            config["lambda"] = 1
        config["train_batch_size"] = int(config["train_batch_size"])
        return config
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--max", type=int, default=1000000)
        parser.add_argument("--algo", type=str, default="PPO")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--num_samples", type=int, default=4)
        parser.add_argument("--t_ready", type=int, default=50000)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--horizon", type=int, default=1600
        )  # make this 1000 for other envs
        parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
        parser.add_argument("--env_name", type=str, default="BipedalWalker-v2")
        parser.add_argument(
            "--criteria", type=str, default="timesteps_total"
        )  # "training_iteration", "time_total_s"
        parser.add_argument(
            "--net", type=str, default="32_32"
        )  # May be important to use a larger network for bigger tasks.
        parser.add_argument("--filename", type=str, default="")
        parser.add_argument("--method", type=str, default="pb2")  # ['pbt', 'pb2']
        parser.add_argument("--save_csv", type=bool, default=False)
    
        args = parser.parse_args()
    
        # bipedalwalker needs 1600
        if args.env_name in ["BipedalWalker-v2", "BipedalWalker-v3"]:
            horizon = 1600
        else:
            horizon = 1000
    
        pbt = PopulationBasedTraining(
            time_attr=args.criteria,
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=args.t_ready,
            resample_probability=args.perturb,
            quantile_fraction=args.perturb,  # copy bottom % with top %
            # Specifies the search space for these hyperparams
            hyperparam_mutations={
                "lambda": lambda: random.uniform(0.9, 1.0),
                "clip_param": lambda: random.uniform(0.1, 0.5),
                "lr": lambda: random.uniform(1e-3, 1e-5),
                "train_batch_size": lambda: random.randint(1000, 60000),
            },
            custom_explore_fn=explore,
        )
    
        pb2 = PB2(
            time_attr=args.criteria,
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=args.t_ready,
            quantile_fraction=args.perturb,  # copy bottom % with top %
            # Specifies the hyperparam search space
            hyperparam_bounds={
                "lambda": [0.9, 1.0],
                "clip_param": [0.1, 0.5],
                "lr": [1e-5, 1e-3],
                "train_batch_size": [1000, 60000],
            },
        )
    
        methods = {"pbt": pbt, "pb2": pb2}
    
        timelog = (
            str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
        )
    
        args.dir = "{}_{}_{}_Size{}_{}_{}".format(
            args.algo,
            args.filename,
            args.method,
            str(args.num_samples),
            args.env_name,
            args.criteria,
        )
    
        analysis = run(
            args.algo,
            name="{}_{}_{}_seed{}_{}".format(
                timelog, args.method, args.env_name, str(args.seed), args.filename
            ),
            scheduler=methods[args.method],
            verbose=1,
            num_samples=args.num_samples,
            reuse_actors=True,
            stop={args.criteria: args.max},
            config={
                "env": args.env_name,
                "log_level": "INFO",
                "seed": args.seed,
                "kl_coeff": 1.0,
                "num_gpus": 0,
                "horizon": horizon,
                "observation_filter": "MeanStdFilter",
                "model": {
                    "fcnet_hiddens": [
                        int(args.net.split("_")[0]),
                        int(args.net.split("_")[1]),
                    ],
                    "free_log_std": True,
                },
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 128,
                "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
                "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
                "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
                "train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
            },
        )
    
        all_dfs = list(analysis.trial_dataframes.values())
    
        results = pd.DataFrame()
        for i in range(args.num_samples):
            df = all_dfs[i]
            df = df[
                [
                    "timesteps_total",
                    "episodes_total",
                    "episode_reward_mean",
                    "info/learner/default_policy/cur_kl_coeff",
                ]
            ]
            df["Agent"] = i
            results = pd.concat([results, df]).reset_index(drop=True)
    
        if args.save_csv:
            if not (os.path.exists("data/" + args.dir)):
                os.makedirs("data/" + args.dir)
    
            results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))

---

## PBT ConvNet Example

# PBT ConvNet Example
    
    
    #!/usr/bin/env python
    
    # __tutorial_imports_begin__
    import argparse
    import os
    
    import numpy as np
    import torch
    import torch.optim as optim
    
    import ray
    from ray import tune
    from ray.tune import Checkpoint
    from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test_func
    from ray.tune.schedulers import PopulationBasedTraining
    
    # __tutorial_imports_end__
    
    
    # __train_begin__
    def train_convnet(config):
        # Create our data loaders, model, and optmizer.
        step = 0
        train_loader, test_loader = get_data_loaders()
        model = ConvNet()
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9),
        )
    
        # If `get_checkpoint()` is not None, then we are resuming from a checkpoint.
        # Load model state and iteration step from checkpoint.
        if tune.get_checkpoint():
            print("Loading from checkpoint.")
            loaded_checkpoint = tune.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint["model"])
                step = checkpoint["step"]
    
        while True:
            ray.tune.examples.mnist_pytorch.train_func(model, optimizer, train_loader)
            acc = test_func(model, test_loader)
            checkpoint = None
            if step % 5 == 0:
                # Every 5 steps, checkpoint our current state.
                # First get the checkpoint directory from tune.
                # Need to create a directory under current working directory
                # to construct checkpoint object from.
                os.makedirs("my_model", exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                    },
                    "my_model/checkpoint.pt",
                )
                checkpoint = Checkpoint.from_directory("my_model")
    
            step += 1
            tune.report({"mean_accuracy": acc}, checkpoint=checkpoint)
    
    
    # __train_end__
    
    
    def eval_best_model(results: tune.ResultGrid):
        """Test the best model given output of tuner.fit()."""
        with results.get_best_result().checkpoint.as_directory() as best_checkpoint_path:
            best_model = ConvNet()
            best_checkpoint = torch.load(
                os.path.join(best_checkpoint_path, "checkpoint.pt")
            )
            best_model.load_state_dict(best_checkpoint["model"])
            # Note that test only runs on a small random set of the test data, thus the
            # accuracy may be different from metrics shown in tuning process.
            test_acc = test_func(best_model, get_data_loaders()[1])
            print("best model accuracy: ", test_acc)
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        # __pbt_begin__
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=5,
            hyperparam_mutations={
                # distribution for resampling
                "lr": lambda: np.random.uniform(0.0001, 1),
                # allow perturbations within this set of categorical values
                "momentum": [0.8, 0.9, 0.99],
            },
        )
    
        # __pbt_end__
    
        # __tune_begin__
        class CustomStopper(tune.Stopper):
            def __init__(self):
                self.should_stop = False
    
            def __call__(self, trial_id, result):
                max_iter = 5 if args.smoke_test else 100
                if not self.should_stop and result["mean_accuracy"] > 0.96:
                    self.should_stop = True
                return self.should_stop or result["training_iteration"] >= max_iter
    
            def stop_all(self):
                return self.should_stop
    
        stopper = CustomStopper()
    
        tuner = tune.Tuner(
            train_convnet,
            run_config=tune.RunConfig(
                name="pbt_test",
                stop=stopper,
                verbose=1,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=4,
                ),
            ),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                metric="mean_accuracy",
                mode="max",
                num_samples=4,
                reuse_actors=True,
            ),
            param_space={
                "lr": tune.uniform(0.001, 1),
                "momentum": tune.uniform(0.001, 1),
            },
        )
        results = tuner.fit()
        # __tune_end__
    
        eval_best_model(results)

---

## PBT Function Example

# PBT Function Example

The following script produces the following results. For a population of 8 trials, the PBT learning rate schedule roughly matches the optimal learning rate schedule.

    #!/usr/bin/env python
    
    import argparse
    import json
    import os
    import random
    import tempfile
    
    import numpy as np
    
    import ray
    from ray import tune
    from ray.tune import Checkpoint
    from ray.tune.schedulers import PopulationBasedTraining
    
    
    def pbt_function(config):
        """Toy PBT problem for benchmarking adaptive learning rate.
    
        The goal is to optimize this trainable's accuracy. The accuracy increases
        fastest at the optimal lr, which is a function of the current accuracy.
    
        The optimal lr schedule for this problem is the triangle wave as follows.
        Note that many lr schedules for real models also follow this shape:
    
         best lr
          ^
          |    /\
          |   /  \
          |  /    \
          | /      \
    
        In this problem, using PBT with a population of 2-4 is sufficient to
        roughly approximate this lr schedule. Higher population sizes will yield
        faster convergence. Training will not converge without PBT.
        """
        lr = config["lr"]
        checkpoint_interval = config.get("checkpoint_interval", 1)
    
        accuracy = 0.0  # end = 1000
    
        # NOTE: See below why step is initialized to 1
        step = 1
        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, "checkpoint.json"), "r") as f:
                    checkpoint_dict = json.load(f)
    
            accuracy = checkpoint_dict["acc"]
            last_step = checkpoint_dict["step"]
            # Current step should be 1 more than the last checkpoint step
            step = last_step + 1
    
        # triangle wave:
        #  - start at 0.001 @ t=0,
        #  - peak at 0.01 @ t=midpoint,
        #  - end at 0.001 @ t=midpoint * 2,
        midpoint = 100  # lr starts decreasing after acc > midpoint
        q_tolerance = 3  # penalize exceeding lr by more than this multiple
        noise_level = 2  # add gaussian noise to the acc increase
    
        # Let `stop={"done": True}` in the configs below handle trial stopping
        while True:
            if accuracy < midpoint:
                optimal_lr = 0.01 * accuracy / midpoint
            else:
                optimal_lr = 0.01 - 0.01 * (accuracy - midpoint) / midpoint
            optimal_lr = min(0.01, max(0.001, optimal_lr))
    
            # compute accuracy increase
            q_err = max(lr, optimal_lr) / min(lr, optimal_lr)
            if q_err < q_tolerance:
                accuracy += (1.0 / q_err) * random.random()
            elif lr > optimal_lr:
                accuracy -= (q_err - q_tolerance) * random.random()
            accuracy += noise_level * np.random.normal()
            accuracy = max(0, accuracy)
    
            metrics = {
                "mean_accuracy": accuracy,
                "cur_lr": lr,
                "optimal_lr": optimal_lr,  # for debugging
                "q_err": q_err,  # for debugging
                "done": accuracy > midpoint * 2,  # this stops the training process
            }
    
            if step % checkpoint_interval == 0:
                # Checkpoint every `checkpoint_interval` steps
                # NOTE: if we initialized `step=0` above, our checkpointing and perturbing
                # would be out of sync by 1 step.
                # Ex: if `checkpoint_interval` = `perturbation_interval` = 3
                # step:                0 (checkpoint)  1     2            3 (checkpoint)
                # training_iteration:  1               2     3 (perturb)  4
                with tempfile.TemporaryDirectory() as tempdir:
                    with open(os.path.join(tempdir, "checkpoint.json"), "w") as f:
                        checkpoint_dict = {"acc": accuracy, "step": step}
                        json.dump(checkpoint_dict, f)
                    tune.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                tune.report(metrics)
            step += 1
    
    
    def run_tune_pbt(smoke_test=False):
        perturbation_interval = 5
        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            hyperparam_mutations={
                # distribution for resampling
                "lr": tune.uniform(0.0001, 0.02),
                # allow perturbations within this set of categorical values
                "some_other_factor": [1, 2],
            },
        )
    
        tuner = tune.Tuner(
            pbt_function,
            run_config=tune.RunConfig(
                name="pbt_function_api_example",
                verbose=False,
                stop={
                    # Stop when done = True or at some # of train steps
                    # (whichever comes first)
                    "done": True,
                    "training_iteration": 10 if smoke_test else 1000,
                },
                failure_config=tune.FailureConfig(
                    fail_fast=True,
                ),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=2,
                ),
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                metric="mean_accuracy",
                mode="max",
                num_samples=8,
                reuse_actors=True,
            ),
            param_space={
                "lr": 0.0001,
                # Note: `some_other_factor` is perturbed because it is specified under
                # the PBT scheduler's `hyperparam_mutations` argument, but has no effect on
                # the model training in this example
                "some_other_factor": 1,
                # Note: `checkpoint_interval` will not be perturbed (since it's not
                # included above), and it will be used to determine how many steps to take
                # between each checkpoint.
                # We recommend matching `perturbation_interval` and `checkpoint_interval`
                # (e.g. checkpoint every 4 steps, and perturb on those same steps)
                # or making `perturbation_interval` a multiple of `checkpoint_interval`
                # (e.g. checkpoint every 2 steps, and perturb every 4 steps).
                # This is to ensure that the lastest checkpoints are being used by PBT
                # when trials decide to exploit. If checkpointing and perturbing are not
                # aligned, then PBT may use a stale checkpoint to resume from.
                "checkpoint_interval": perturbation_interval,
            },
        )
        results = tuner.fit()
    
        print("Best hyperparameters found were: ", results.get_best_result().config)
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test",
            action="store_true",
            default=False,
            help="Finish quickly for testing",
        )
        args, _ = parser.parse_known_args()
        if args.smoke_test:
            ray.init(num_cpus=2)  # force pausing to happen for test
    
        run_tune_pbt(smoke_test=args.smoke_test)

---

## Memory NN Example

# Memory NN Example
    
    
    """Example training a memory neural net on the bAbI dataset.
    
    References Keras and is based off of https://keras.io/examples/babi_memnn/.
    """
    
    from __future__ import print_function
    
    import argparse
    import os
    import re
    import sys
    import tarfile
    
    import numpy as np
    from filelock import FileLock
    
    from ray import tune
    
    if sys.version_info >= (3, 12):
        # Skip this test in Python 3.12+ because TensorFlow is not supported.
        sys.exit(0)
    else:
        from tensorflow.keras.layers import (
            LSTM,
            Activation,
            Dense,
            Dropout,
            Embedding,
            Input,
            Permute,
            add,
            concatenate,
            dot,
        )
        from tensorflow.keras.models import Model, Sequential, load_model
        from tensorflow.keras.optimizers import RMSprop
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import get_file
    
    
    def tokenize(sent):
        """Return the tokens of a sentence including punctuation.
    
        >>> tokenize("Bob dropped the apple. Where is the apple?")
        ["Bob", "dropped", "the", "apple", ".", "Where", "is", "the", "apple", "?"]
        """
        return [x.strip() for x in re.split(r"(\W+)?", sent) if x and x.strip()]
    
    
    def parse_stories(lines, only_supporting=False):
        """Parse stories provided in the bAbi tasks format
    
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        data = []
        story = []
        for line in lines:
            line = line.decode("utf-8").strip()
            nid, line = line.split(" ", 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if "\t" in line:
                q, a, supporting = line.split("\t")
                q = tokenize(q)
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append("")
            else:
                sent = tokenize(line)
                story.append(sent)
        return data
    
    
    def get_stories(f, only_supporting=False, max_length=None):
        """Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
    
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        """
    
        def flatten(data):
            return sum(data, [])
    
        data = parse_stories(f.readlines(), only_supporting=only_supporting)
        data = [
            (flatten(story), q, answer)
            for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length
        ]
        return data
    
    
    def vectorize_stories(word_idx, story_maxlen, query_maxlen, data):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([word_idx[w] for w in story])
            queries.append([word_idx[w] for w in query])
            answers.append(word_idx[answer])
        return (
            pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers),
        )
    
    
    def read_data(finish_fast=False):
        # Get the file
        try:
            path = get_file(
                "babi-tasks-v1-2.tar.gz",
                origin="https://s3.amazonaws.com/text-datasets/"
                "babi_tasks_1-20_v1-2.tar.gz",
            )
        except Exception:
            print(
                "Error downloading dataset, please download it manually:\n"
                "$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2"  # noqa: E501
                ".tar.gz\n"
                "$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz"  # noqa: E501
            )
            raise
    
        # Choose challenge
        challenges = {
            # QA1 with 10,000 samples
            "single_supporting_fact_10k": "tasks_1-20_v1-2/en-10k/qa1_"
            "single-supporting-fact_{}.txt",
            # QA2 with 10,000 samples
            "two_supporting_facts_10k": "tasks_1-20_v1-2/en-10k/qa2_"
            "two-supporting-facts_{}.txt",
        }
        challenge_type = "single_supporting_fact_10k"
        challenge = challenges[challenge_type]
    
        with tarfile.open(path) as tar:
            train_stories = get_stories(tar.extractfile(challenge.format("train")))
            test_stories = get_stories(tar.extractfile(challenge.format("test")))
        if finish_fast:
            train_stories = train_stories[:64]
            test_stories = test_stories[:64]
        return train_stories, test_stories
    
    
    class MemNNModel(tune.Trainable):
        def build_model(self):
            """Helper method for creating the model"""
            vocab = set()
            for story, q, answer in self.train_stories + self.test_stories:
                vocab |= set(story + q + [answer])
            vocab = sorted(vocab)
    
            # Reserve 0 for masking via pad_sequences
            vocab_size = len(vocab) + 1
            story_maxlen = max(len(x) for x, _, _ in self.train_stories + self.test_stories)
            query_maxlen = max(len(x) for _, x, _ in self.train_stories + self.test_stories)
    
            word_idx = {c: i + 1 for i, c in enumerate(vocab)}
            self.inputs_train, self.queries_train, self.answers_train = vectorize_stories(
                word_idx, story_maxlen, query_maxlen, self.train_stories
            )
            self.inputs_test, self.queries_test, self.answers_test = vectorize_stories(
                word_idx, story_maxlen, query_maxlen, self.test_stories
            )
    
            # placeholders
            input_sequence = Input((story_maxlen,))
            question = Input((query_maxlen,))
    
            # encoders
            # embed the input sequence into a sequence of vectors
            input_encoder_m = Sequential()
            input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
            input_encoder_m.add(Dropout(self.config.get("dropout", 0.3)))
            # output: (samples, story_maxlen, embedding_dim)
    
            # embed the input into a sequence of vectors of size query_maxlen
            input_encoder_c = Sequential()
            input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
            input_encoder_c.add(Dropout(self.config.get("dropout", 0.3)))
            # output: (samples, story_maxlen, query_maxlen)
    
            # embed the question into a sequence of vectors
            question_encoder = Sequential()
            question_encoder.add(
                Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen)
            )
            question_encoder.add(Dropout(self.config.get("dropout", 0.3)))
            # output: (samples, query_maxlen, embedding_dim)
    
            # encode input sequence and questions (which are indices)
            # to sequences of dense vectors
            input_encoded_m = input_encoder_m(input_sequence)
            input_encoded_c = input_encoder_c(input_sequence)
            question_encoded = question_encoder(question)
    
            # compute a "match" between the first input vector sequence
            # and the question vector sequence
            # shape: `(samples, story_maxlen, query_maxlen)`
            match = dot([input_encoded_m, question_encoded], axes=(2, 2))
            match = Activation("softmax")(match)
    
            # add the match matrix with the second input vector sequence
            response = add(
                [match, input_encoded_c]
            )  # (samples, story_maxlen, query_maxlen)
            response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
    
            # concatenate the match matrix with the question vector sequence
            answer = concatenate([response, question_encoded])
    
            # the original paper uses a matrix multiplication.
            # we choose to use a RNN instead.
            answer = LSTM(32)(answer)  # (samples, 32)
    
            # one regularization layer -- more would probably be needed.
            answer = Dropout(self.config.get("dropout", 0.3))(answer)
            answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
            # we output a probability distribution over the vocabulary
            answer = Activation("softmax")(answer)
    
            # build the final model
            model = Model([input_sequence, question], answer)
            return model
    
        def setup(self, config):
            with FileLock(os.path.expanduser("~/.tune.lock")):
                self.train_stories, self.test_stories = read_data(config["finish_fast"])
            model = self.build_model()
            rmsprop = RMSprop(
                lr=self.config.get("lr", 1e-3), rho=self.config.get("rho", 0.9)
            )
            model.compile(
                optimizer=rmsprop,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.model = model
    
        def step(self):
            # train
            self.model.fit(
                [self.inputs_train, self.queries_train],
                self.answers_train,
                batch_size=self.config.get("batch_size", 32),
                epochs=self.config.get("epochs", 1),
                validation_data=([self.inputs_test, self.queries_test], self.answers_test),
                verbose=0,
            )
            _, accuracy = self.model.evaluate(
                [self.inputs_train, self.queries_train], self.answers_train, verbose=0
            )
            return {"mean_accuracy": accuracy}
    
        def save_checkpoint(self, checkpoint_dir):
            file_path = checkpoint_dir + "/model"
            self.model.save(file_path)
    
        def load_checkpoint(self, checkpoint_dir):
            # See https://stackoverflow.com/a/42763323
            del self.model
            file_path = checkpoint_dir + "/model"
            self.model = load_model(file_path)
    
    
    if __name__ == "__main__":
        import ray
        from ray.tune.schedulers import PopulationBasedTraining
    
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        if args.smoke_test:
            ray.init(num_cpus=2)
    
        perturbation_interval = 2
        pbt = PopulationBasedTraining(
            perturbation_interval=perturbation_interval,
            hyperparam_mutations={
                "dropout": lambda: np.random.uniform(0, 1),
                "lr": lambda: 10 ** np.random.randint(-10, 0),
                "rho": lambda: np.random.uniform(0, 1),
            },
        )
    
        tuner = tune.Tuner(
            MemNNModel,
            run_config=tune.RunConfig(
                name="pbt_babi_memnn",
                stop={"training_iteration": 4 if args.smoke_test else 100},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=perturbation_interval,
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=2,
                ),
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                metric="mean_accuracy",
                mode="max",
                num_samples=2,
                reuse_actors=True,
            ),
            param_space={
                "finish_fast": args.smoke_test,
                "batch_size": 32,
                "epochs": 1,
                "dropout": 0.3,
                "lr": 0.01,
                "rho": 0.9,
            },
        )
        tuner.fit()

---

## Keras Cifar10 Example

# Keras Cifar10 Example
    
    
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    """Train keras CNN on the CIFAR10 small images dataset.
    
    The model comes from: https://zhuanlan.zhihu.com/p/29214791,
    and it gets to about 87% validation accuracy in 100 epochs.
    
    Note that the script requires a machine with 4 GPUs. You
    can set {"gpu": 0} to use CPUs for training, although
    it is less efficient.
    """
    
    from __future__ import print_function
    
    import argparse
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.layers import (
        Convolution2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    from ray import tune
    from ray.tune import Trainable
    from ray.tune.schedulers import PopulationBasedTraining
    
    num_classes = 10
    NUM_SAMPLES = 128
    
    
    class Cifar10Model(Trainable):
        def _read_data(self):
            # The data, split between train and test sets:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
            # Convert class vectors to binary class matrices.
            y_train = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
            x_train = x_train.astype("float32")
            x_train /= 255
            x_test = x_test.astype("float32")
            x_test /= 255
    
            return (x_train, y_train), (x_test, y_test)
    
        def _build_model(self, input_shape):
            x = Input(shape=(32, 32, 3))
            y = x
            y = Convolution2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = Convolution2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)
    
            y = Convolution2D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = Convolution2D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)
    
            y = Convolution2D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = Convolution2D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)
    
            y = Flatten()(y)
            y = Dropout(self.config.get("dropout", 0.5))(y)
            y = Dense(units=10, activation="softmax", kernel_initializer="he_normal")(y)
    
            model = Model(inputs=x, outputs=y, name="model1")
            return model
    
        def setup(self, config):
            self.train_data, self.test_data = self._read_data()
            x_train = self.train_data[0]
            model = self._build_model(x_train.shape[1:])
    
            opt = tf.keras.optimizers.Adadelta(
                lr=self.config.get("lr", 1e-4), weight_decay=self.config.get("decay", 1e-4)
            )
            model.compile(
                loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            self.model = model
    
        def step(self):
            x_train, y_train = self.train_data
            x_train, y_train = x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
            x_test, y_test = self.test_data
            x_test, y_test = x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]
    
            aug_gen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by dataset std
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (degrees, 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
            )
    
            aug_gen.fit(x_train)
            batch_size = self.config.get("batch_size", 64)
            gen = aug_gen.flow(x_train, y_train, batch_size=batch_size)
            self.model.fit_generator(
                generator=gen, epochs=self.config.get("epochs", 1), validation_data=None
            )
    
            # loss, accuracy
            _, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            return {"mean_accuracy": accuracy}
    
        def save_checkpoint(self, checkpoint_dir):
            file_path = checkpoint_dir + "/model"
            self.model.save(file_path)
    
        def load_checkpoint(self, checkpoint_dir):
            # See https://stackoverflow.com/a/42763323
            del self.model
            file_path = checkpoint_dir + "/model"
            self.model = load_model(file_path)
    
        def cleanup(self):
            # If need, save your model when exit.
            # saved_path = self.model.save(self.logdir)
            # print("save model at: ", saved_path)
            pass
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        space = {
            "epochs": 1,
            "batch_size": 64,
            "lr": tune.grid_search([10**-4, 10**-5]),
            "decay": tune.sample_from(lambda spec: spec.config.lr / 100.0),
            "dropout": tune.grid_search([0.25, 0.5]),
        }
        if args.smoke_test:
            space["lr"] = 10**-4
            space["dropout"] = 0.5
    
        perturbation_interval = 10
        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            hyperparam_mutations={
                "dropout": lambda _: np.random.uniform(0, 1),
            },
        )
    
        tuner = tune.Tuner(
            tune.with_resources(
                Cifar10Model,
                resources={"cpu": 1, "gpu": 1},
            ),
            run_config=tune.RunConfig(
                name="pbt_cifar10",
                stop={
                    "mean_accuracy": 0.80,
                    "training_iteration": 30,
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=perturbation_interval,
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=2,
                ),
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=4,
                metric="mean_accuracy",
                mode="max",
                reuse_actors=True,
            ),
            param_space=space,
        )
        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)

---

## TensorFlow MNIST Example

# TensorFlow MNIST Example
    
    
    #!/usr/bin/env python
    # coding: utf-8
    #
    # This example showcases how to use TF2.0 APIs with Tune.
    # Original code: https://www.tensorflow.org/tutorials/quickstart/advanced
    #
    # As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
    # functionality does not interact nicely with Ray actors. One way to get around
    # this is to `import tensorflow` inside the Tune Trainable.
    #
    
    import argparse
    import os
    import sys
    
    from filelock import FileLock
    
    from ray import tune
    
    MAX_TRAIN_BATCH = 10
    
    if sys.version_info >= (3, 12):
        # Tensorflow is not installed for Python 3.12 because of keras compatibility.
        sys.exit(0)
    else:
        from tensorflow.keras import Model
        from tensorflow.keras.datasets.mnist import load_data
        from tensorflow.keras.layers import Conv2D, Dense, Flatten
    
    
    class MyModel(Model):
        def __init__(self, hiddens=128):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation="relu")
            self.flatten = Flatten()
            self.d1 = Dense(hiddens, activation="relu")
            self.d2 = Dense(10, activation="softmax")
    
        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)
    
    
    class MNISTTrainable(tune.Trainable):
        def setup(self, config):
            # IMPORTANT: See the above note.
            import tensorflow as tf
    
            # Use FileLock to avoid race conditions.
            with FileLock(os.path.expanduser("~/.tune.lock")):
                (x_train, y_train), (x_test, y_test) = load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
    
            # Add a channels dimension
            x_train = x_train[..., tf.newaxis]
            x_test = x_test[..., tf.newaxis]
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            self.train_ds = self.train_ds.shuffle(10000).batch(config.get("batch", 32))
    
            self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
            self.model = MyModel(hiddens=config.get("hiddens", 128))
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam()
            self.train_loss = tf.keras.metrics.Mean(name="train_loss")
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name="train_accuracy"
            )
    
            self.test_loss = tf.keras.metrics.Mean(name="test_loss")
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name="test_accuracy"
            )
    
            @tf.function
            def train_step(images, labels):
                with tf.GradientTape() as tape:
                    predictions = self.model(images)
                    loss = self.loss_object(labels, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
    
                self.train_loss(loss)
                self.train_accuracy(labels, predictions)
    
            @tf.function
            def test_step(images, labels):
                predictions = self.model(images)
                t_loss = self.loss_object(labels, predictions)
    
                self.test_loss(t_loss)
                self.test_accuracy(labels, predictions)
    
            self.tf_train_step = train_step
            self.tf_test_step = test_step
    
        def save_checkpoint(self, checkpoint_dir: str):
            return None
    
        def load_checkpoint(self, checkpoint):
            return None
    
        def step(self):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
    
            for idx, (images, labels) in enumerate(self.train_ds):
                if idx > MAX_TRAIN_BATCH:  # This is optional and can be removed.
                    break
                self.tf_train_step(images, labels)
    
            for test_images, test_labels in self.test_ds:
                self.tf_test_step(test_images, test_labels)
    
            # It is important to return tf.Tensors as numpy objects.
            return {
                "epoch": self.iteration,
                "loss": self.train_loss.result().numpy(),
                "accuracy": self.train_accuracy.result().numpy() * 100,
                "test_loss": self.test_loss.result().numpy(),
                "mean_accuracy": self.test_accuracy.result().numpy() * 100,
            }
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        tuner = tune.Tuner(
            MNISTTrainable,
            tune_config=tune.TuneConfig(
                metric="test_loss",
                mode="min",
            ),
            run_config=tune.RunConfig(
                stop={"training_iteration": 5 if args.smoke_test else 50},
                verbose=1,
            ),
            param_space={"hiddens": tune.grid_search([32, 64, 128])},
        )
        results = tuner.fit()
    
        print("Best hyperparameters found were: ", results.get_best_result().config)

---

## tune_basic_example

# tune_basic_example
    
    
    """This example demonstrates basic Ray Tune random search and grid search."""
    import time
    
    import ray
    from ray import tune
    
    
    def evaluation_fn(step, width, height):
        time.sleep(0.1)
        return (0.1 + width * step / 100) ** (-1) + height * 0.1
    
    
    def easy_objective(config):
        # Hyperparameters
        width, height = config["width"], config["height"]
    
        for step in range(config["steps"]):
            # Iterative training function - can be any arbitrary training procedure
            intermediate_score = evaluation_fn(step, width, height)
            # Feed the score back back to Tune.
            tune.report({"iterations": step, "mean_loss": intermediate_score})
    
    
    if __name__ == "__main__":
        import argparse
    
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing"
        )
        args, _ = parser.parse_known_args()
    
        ray.init(configure_logging=False)
    
        # This will do a grid search over the `activation` parameter. This means
        # that each of the two values (`relu` and `tanh`) will be sampled once
        # for each sample (`num_samples`). We end up with 2 * 50 = 100 samples.
        # The `width` and `height` parameters are sampled randomly.
        # `steps` is a constant parameter.
    
        tuner = tune.Tuner(
            easy_objective,
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                num_samples=5 if args.smoke_test else 50,
            ),
            param_space={
                "steps": 5 if args.smoke_test else 100,
                "width": tune.uniform(0, 20),
                "height": tune.uniform(-100, 100),
                "activation": tune.grid_search(["relu", "tanh"]),
            },
        )
        results = tuner.fit()
    
        print("Best hyperparameters found were: ", results.get_best_result().config)

---

## XGBoost Dynamic Resources Example

# XGBoost Dynamic Resources Example
    
    
    from typing import TYPE_CHECKING, Any, Dict, Optional
    
    import sklearn.datasets
    import sklearn.metrics
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
    import ray
    from ray import tune
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    from ray.tune.experiment import Trial
    from ray.tune.integration.xgboost import TuneReportCheckpointCallback
    from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler
    
    if TYPE_CHECKING:
        from ray.tune.execution.tune_controller import TuneController
    
    CHECKPOINT_FILENAME = "booster-checkpoint.json"
    
    
    def get_best_model_checkpoint(best_result: "ray.tune.Result"):
        best_bst = TuneReportCheckpointCallback.get_model(
            best_result.checkpoint, filename=CHECKPOINT_FILENAME
        )
    
        accuracy = 1.0 - best_result.metrics["eval-logloss"]
        print(f"Best model parameters: {best_result.config}")
        print(f"Best model total accuracy: {accuracy:.4f}")
        return best_bst
    
    
    # our train function needs to be able to checkpoint
    # to work with ResourceChangingScheduler
    def train_breast_cancer(config: dict):
        # This is a simple training function to be passed into Tune
        # Load dataset
        data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
        # Split into train and test set
        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
        # Build input matrices for XGBoost
        train_set = xgb.DMatrix(train_x, label=train_y)
        test_set = xgb.DMatrix(test_x, label=test_y)
    
        # Checkpointing needs to be set up in order for dynamic
        # resource allocation to work as intended
        xgb_model = None
        checkpoint = tune.get_checkpoint()
        if checkpoint:
            xgb_model = TuneReportCheckpointCallback.get_model(
                checkpoint, filename=CHECKPOINT_FILENAME
            )
    
        # Set `nthread` to the number of CPUs available to the trial,
        # which is assigned by the scheduler.
        config["nthread"] = int(tune.get_context().get_trial_resources().head_cpus)
        print(f"nthreads: {config['nthread']} xgb_model: {xgb_model}")
        # Train the classifier, using the Tune callback
        xgb.train(
            config,
            train_set,
            evals=[(test_set, "eval")],
            verbose_eval=False,
            xgb_model=xgb_model,
            callbacks=[
                TuneReportCheckpointCallback(
                    # checkpointing should happen every iteration
                    # with dynamic resource allocation
                    frequency=1,
                    filename=CHECKPOINT_FILENAME,
                )
            ],
        )
    
    
    def tune_xgboost():
        search_space = {
            # You can mix constants with search space objects.
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
            "max_depth": 9,
            "learning_rate": 1,
            "min_child_weight": tune.grid_search([2, 3]),
            "subsample": tune.grid_search([0.8, 0.9]),
            "colsample_bynode": tune.grid_search([0.8, 0.9]),
            "random_state": 1,
            "num_parallel_tree": 2000,
        }
        # This will enable aggressive early stopping of bad trials.
        base_scheduler = ASHAScheduler(
            max_t=16, grace_period=1, reduction_factor=2  # 16 training iterations
        )
    
        def example_resources_allocation_function(
            tune_controller: "TuneController",
            trial: Trial,
            result: Dict[str, Any],
            scheduler: "ResourceChangingScheduler",
        ) -> Optional[PlacementGroupFactory]:
            """This is a basic example of a resource allocating function.
    
            The function naively balances available CPUs over live trials.
    
            This function returns a new ``PlacementGroupFactory`` with updated
            resource requirements, or None. If the returned
            ``PlacementGroupFactory`` is equal by value to the one the
            trial has currently, the scheduler will skip the update process
            internally (same with None).
    
            See :class:`DistributeResources` for a more complex,
            robust approach.
    
            Args:
                tune_controller: Trial runner for this Tune run.
                    Can be used to obtain information about other trials.
                trial: The trial to allocate new resources to.
                result: The latest results of trial.
                scheduler: The scheduler calling the function.
            """
    
            # Get base trial resources as defined in
            # ``tune.with_resources``
            base_trial_resource = scheduler._base_trial_resources
    
            # Don't bother if this is just the first iteration
            if result["training_iteration"] < 1:
                return None
    
            # default values if resources_per_trial is unspecified
            if base_trial_resource is None:
                base_trial_resource = PlacementGroupFactory([{"CPU": 1, "GPU": 0}])
    
            # Assume that the number of CPUs cannot go below what was
            # specified in ``Tuner.fit()``.
            min_cpu = base_trial_resource.required_resources.get("CPU", 0)
    
            # Get the number of CPUs available in total (not just free)
            total_available_cpus = tune_controller._resource_updater.get_num_cpus()
    
            # Divide the free CPUs among all live trials
            cpu_to_use = max(
                min_cpu, total_available_cpus // len(tune_controller.get_live_trials())
            )
    
            # Assign new CPUs to the trial in a PlacementGroupFactory
            return PlacementGroupFactory([{"CPU": cpu_to_use, "GPU": 0}])
    
        # You can either define your own resources_allocation_function, or
        # use the default one - DistributeResources
    
        # from ray.tune.schedulers.resource_changing_scheduler import \
        #    DistributeResources
    
        scheduler = ResourceChangingScheduler(
            base_scheduler=base_scheduler,
            resources_allocation_function=example_resources_allocation_function,
            # resources_allocation_function=DistributeResources()  # default
        )
    
        tuner = tune.Tuner(
            tune.with_resources(
                train_breast_cancer, resources=PlacementGroupFactory([{"CPU": 1, "GPU": 0}])
            ),
            tune_config=tune.TuneConfig(
                metric="eval-logloss",
                mode="min",
                num_samples=1,
                scheduler=scheduler,
            ),
            param_space=search_space,
        )
        results = tuner.fit()
    
        return results.get_best_result()
    
    
    if __name__ == "__main__":
        ray.init(num_cpus=8)
    
        best_result = tune_xgboost()
        best_bst = get_best_model_checkpoint(best_result)
    
        # You could now do further predictions with
        # best_bst.predict(...)

---

