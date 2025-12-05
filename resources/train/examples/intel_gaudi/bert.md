# BERT Model Training with Intel Gaudi

---

# BERT Model Training with Intel Gaudi

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=intel_gaudi-bert)   

In this notebook, we will train a BERT model for sequence classification using the Yelp review full dataset. We will use the `transformers` and `datasets` libraries from Hugging Face, along with `ray.train` for distributed training.

[Intel Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Intel Habana Labs. For more information, see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/).

## Configuration

A node with Gaudi/Gaudi2 installed is required to run this example. Both Gaudi and Gaudi2 have 8 HPUs. We will use 2 workers to train the model, each using 1 HPU.

We recommend using a prebuilt container to run these examples. To run a container, you need Docker. See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

Next, follow [Run Using Containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install the Gaudi drivers and container runtime.

Next, start the Gaudi container:
    
    
    docker pull vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    

Inside the container, install the following dependencies to run this notebook.
    
    
    pip install ray[train] notebook transformers datasets evaluate
    
    
    
    # Import necessary libraries
    
    import os
    from typing import Dict
    
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    import numpy as np
    import evaluate
    from datasets import load_dataset
    import transformers
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    
    import ray.train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.torch import TorchConfig
    from ray.runtime_env import RuntimeEnv
    
    import habana_frameworks.torch.core as htcore
    
    
    
    /usr/local/lib/python3.10/dist-packages/torch/distributed/distributed_c10d.py:252: UserWarning: Device capability of hccl unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.
      warnings.warn(
    

## Metrics Setup

We will use accuracy as our evaluation metric. The `compute_metrics` function will calculate the accuracy of our model’s predictions.
    
    
    # Metrics
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    

## Training Function

This function will be executed by each worker during training. It handles data loading, tokenization, model initialization, and the training loop. Compared to a training function for GPU, no changes are needed to port to HPU. Internally, Ray Train does these things:

  * Detect HPU and set the device.

  * Initializes the habana PyTorch backend.

  * Initializes the habana distributed backend.

    
    
    def train_func_per_worker(config: Dict):
        
        # Datasets
        dataset = load_dataset("yelp_review_full")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
    
        lr = config["lr"]
        epochs = config["epochs"]
        batch_size = config["batch_size_per_worker"]
    
        train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
        eval_dataset = dataset["test"].select(range(1000)).map(tokenize_function, batched=True)
    
        # Prepare dataloader for each worker
        dataloaders = {}
        dataloaders["train"] = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=transformers.default_data_collator, 
            batch_size=batch_size
        )
        dataloaders["test"] = torch.utils.data.DataLoader(
            eval_dataset, 
            shuffle=True, 
            collate_fn=transformers.default_data_collator, 
            batch_size=batch_size
        )
    
        # Obtain HPU device automatically
        device = ray.train.torch.get_device()
    
        # Prepare model and optimizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=5
        )
        model = model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
        # Start training loops
        for epoch in range(epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
    
                # breakpoint()
                for batch  in dataloaders[phase]:
                    batch = {k: v.to(device) for k, v in batch.items()}
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        
                        outputs = model(**batch)
                        loss = outputs.loss
    
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            print(f"train epoch:[{epoch}]\tloss:{loss:.6f}")
    

## Main Training Function

The `train_bert` function sets up the distributed training environment using Ray and starts the training process. To enable training using HPU, we only need to make the following changes:

  * Require an HPU for each worker in ScalingConfig

  * Set backend to “hccl” in TorchConfig

    
    
    def train_bert(num_workers=2):
        global_batch_size = 8
    
        train_config = {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size_per_worker": global_batch_size // num_workers,
        }
    
        # Configure computation resources
        # In ScalingConfig, require an HPU for each worker
        scaling_config = ScalingConfig(num_workers=num_workers, resources_per_worker={"CPU": 1, "HPU": 1})
        # Set backend to hccl in TorchConfig
        torch_config = TorchConfig(backend = "hccl")
        
        # start your ray cluster
        ray.init()
        
        # Initialize a Ray TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func_per_worker,
            train_loop_config=train_config,
            torch_config=torch_config,
            scaling_config=scaling_config,
        )
    
        result = trainer.fit()
        print(f"Training result: {result}")
    

## Start Training

Finally, we call the `train_bert` function to start the training process. You can adjust the number of workers to use.

Note: the following warning is fine, and is resolved in SynapseAI version 1.14.0+:
    
    
    /usr/local/lib/python3.10/dist-packages/torch/distributed/distributed_c10d.py:252: UserWarning: Device capability of hccl unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.
    
    
    
    train_bert(num_workers=2)
    

## Possible outputs
    
    
    Downloading builder script: 100%|██████████| 4.20k/4.20k [00:00<00:00, 27.0MB/s]
    2025-03-03 03:37:08,776 INFO worker.py:1841 -- Started a local Ray instance.
    /usr/local/lib/python3.10/dist-packages/ray/tune/impl/tuner_internal.py:125: RayDeprecationWarning: The `RunConfig` class should be imported from `ray.tune` when passing it to the Tuner. Please update your imports. See this issue for more context and migration options: https://github.com/ray-project/ray/issues/49454. Disable these warnings by setting the environment variable: RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
      _log_deprecation_warning(
    (RayTrainWorker pid=75123) Setting up process group for: env:// [rank=0, world_size=2]
    (TorchTrainer pid=74734) Started distributed worker processes: 
    (TorchTrainer pid=74734) - (node_id=eef984cd0cd96cce50bad1b1dab12e19c809047f10be3c829524a3d1, ip=100.83.111.228, pid=75123) world_rank=0, local_rank=0, node_rank=0
    (TorchTrainer pid=74734) - (node_id=eef984cd0cd96cce50bad1b1dab12e19c809047f10be3c829524a3d1, ip=100.83.111.228, pid=75122) world_rank=1, local_rank=1, node_rank=0
    Generating train split:   0%|          | 0/650000 [00:00<?, ? examples/s]
    Generating train split:   7%|▋         | 45000/650000 [00:00<00:01, 435976.18 examples/s]
    Generating train split:  15%|█▍        | 95000/650000 [00:00<00:01, 469481.51 examples/s]
    Generating train split:  23%|██▎       | 150000/650000 [00:00<00:01, 477676.99 examples/s]
    Generating train split:  31%|███       | 203000/650000 [00:00<00:00, 493746.70 examples/s]
    Generating train split:  43%|████▎     | 279000/650000 [00:00<00:00, 499340.09 examples/s]
    Generating train split:  55%|█████▍    | 355000/650000 [00:00<00:00, 498613.65 examples/s]
    Generating train split:  66%|██████▋   | 431000/650000 [00:00<00:00, 497799.19 examples/s]
    Generating train split:  78%|███████▊  | 506000/650000 [00:01<00:00, 495696.93 examples/s]
    Generating train split:  86%|████████▌ | 556000/650000 [00:01<00:00, 494508.05 examples/s]
    Generating train split:  94%|█████████▎| 609000/650000 [00:01<00:00, 490725.53 examples/s]
    Generating train split: 100%|██████████| 650000/650000 [00:01<00:00, 494916.42 examples/s]
    Generating test split:   0%|          | 0/50000 [00:00<?, ? examples/s]
    Generating test split: 100%|██████████| 50000/50000 [00:00<00:00, 509619.87 examples/s]
    Map:   0%|          | 0/1000 [00:00<?, ? examples/s]
    Map: 100%|██████████| 1000/1000 [00:00<00:00, 3998.33 examples/s]
    Map: 100%|██████████| 1000/1000 [00:00<00:00, 4051.80 examples/s]
    Map: 100%|██████████| 1000/1000 [00:00<00:00, 3869.20 examples/s]
    (RayTrainWorker pid=75123) Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    (RayTrainWorker pid=75123) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Map:   0%|          | 0/1000 [00:00<?, ? examples/s] [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
    Map: 100%|██████████| 1000/1000 [00:00<00:00, 3782.66 examples/s] [repeated 2x across cluster]
    (RayTrainWorker pid=75123) ============================= HABANA PT BRIDGE CONFIGURATION =========================== 
    (RayTrainWorker pid=75123)  PT_HPU_LAZY_MODE = 1
    (RayTrainWorker pid=75123)  PT_HPU_RECIPE_CACHE_CONFIG = ,false,1024
    (RayTrainWorker pid=75123)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
    (RayTrainWorker pid=75123)  PT_HPU_LAZY_ACC_PAR_MODE = 1
    (RayTrainWorker pid=75123)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
    (RayTrainWorker pid=75123)  PT_HPU_EAGER_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=75123)  PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=75123)  PT_HPU_ENABLE_LAZY_COLLECTIVES = 0
    (RayTrainWorker pid=75123) ---------------------------: System Configuration :---------------------------
    (RayTrainWorker pid=75123) Num CPU Cores : 160
    (RayTrainWorker pid=75123) CPU RAM       : 1056374420 KB
    (RayTrainWorker pid=75123) ------------------------------------------------------------------------------
    2025-03-03 03:41:04,658 INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/root/ray_results/TorchTrainer_2025-03-03_03-37-11' in 0.0020s.
    
    View detailed results here: /root/ray_results/TorchTrainer_2025-03-03_03-37-11
    To visualize your results with TensorBoard, run: `tensorboard --logdir /tmp/ray/session_2025-03-03_03-37-06_983992_65223/artifacts/2025-03-03_03-37-11/TorchTrainer_2025-03-03_03-37-11/driver_artifacts`
    
    Training started with configuration:
    ╭─────────────────────────────────────────────────╮
    │ Training config                                 │
    ├─────────────────────────────────────────────────┤
    │ train_loop_config/batch_size_per_worker       4 │
    │ train_loop_config/epochs                     10 │
    │ train_loop_config/lr                      0.001 │
    ╰─────────────────────────────────────────────────╯
    (RayTrainWorker pid=75123) train epoch:[0]      loss:1.979938
    (RayTrainWorker pid=75123) train epoch:[0]      loss:1.756611 [repeated 36x across cluster]
    (RayTrainWorker pid=75123) train epoch:[0]      loss:1.643875 [repeated 180x across cluster]
    (RayTrainWorker pid=75123) train epoch:[0]      loss:1.416416 [repeated 177x across cluster]
    (RayTrainWorker pid=75123) train epoch:[1]      loss:1.272513 [repeated 107x across cluster]
    (RayTrainWorker pid=75123) 
    (RayTrainWorker pid=75123) train epoch:[1]      loss:2.086884 [repeated 155x across cluster]
    (RayTrainWorker pid=75123) train epoch:[1]      loss:1.426217 [repeated 178x across cluster]
    (RayTrainWorker pid=75122) train epoch:[1]      loss:0.991381 [repeated 160x across cluster]
    (RayTrainWorker pid=75123) train epoch:[2]      loss:1.294097 [repeated 28x across cluster]
    (RayTrainWorker pid=75123) train epoch:[2]      loss:1.386306 [repeated 169x across cluster]
    (RayTrainWorker pid=75123) train epoch:[2]      loss:1.190416 [repeated 181x across cluster]
    (RayTrainWorker pid=75123) train epoch:[3]      loss:1.171733 [repeated 130x across cluster]
    (RayTrainWorker pid=75123) train epoch:[3]      loss:1.287821 [repeated 152x across cluster]
    (RayTrainWorker pid=75123) train epoch:[3]      loss:1.055692 [repeated 179x across cluster]
    (RayTrainWorker pid=75122) train epoch:[3]      loss:1.677789 [repeated 162x across cluster]
    (RayTrainWorker pid=75123) train epoch:[4]      loss:0.942071 [repeated 19x across cluster]
    (RayTrainWorker pid=75123) train epoch:[4]      loss:1.592500 [repeated 167x across cluster]
    (RayTrainWorker pid=75123) train epoch:[4]      loss:0.936934 [repeated 180x across cluster]
    (RayTrainWorker pid=75123) 
    (RayTrainWorker pid=75123) train epoch:[5]      loss:2.465384 [repeated 141x across cluster]
    (RayTrainWorker pid=75123) train epoch:[5]      loss:1.659170 [repeated 156x across cluster]
    (RayTrainWorker pid=75123) train epoch:[5]      loss:1.850438 [repeated 180x across cluster]
    (RayTrainWorker pid=75122) train epoch:[5]      loss:1.101623 [repeated 160x across cluster]
    (RayTrainWorker pid=75123) train epoch:[6]      loss:2.125591 [repeated 18x across cluster]
    (RayTrainWorker pid=75123) train epoch:[6]      loss:1.612838 [repeated 170x across cluster]
    (RayTrainWorker pid=75123) train epoch:[6]      loss:1.759160 [repeated 177x across cluster]
    (RayTrainWorker pid=75123) train epoch:[7]      loss:1.338552 [repeated 139x across cluster]
    (RayTrainWorker pid=75123) train epoch:[7]      loss:1.467959 [repeated 157x across cluster]
    (RayTrainWorker pid=75123) train epoch:[7]      loss:1.682137 [repeated 181x across cluster]
    (RayTrainWorker pid=75123) 
    (RayTrainWorker pid=75123) train epoch:[8]      loss:1.395805 [repeated 162x across cluster]
    (RayTrainWorker pid=75123) train epoch:[8]      loss:1.527835 [repeated 153x across cluster]
    (RayTrainWorker pid=75123) train epoch:[8]      loss:1.672311 [repeated 177x across cluster]
    (RayTrainWorker pid=75123) 
    (RayTrainWorker pid=75122) train epoch:[8]      loss:1.093186 [repeated 166x across cluster]
    (RayTrainWorker pid=75123) train epoch:[9]      loss:1.457587 [repeated 13x across cluster]
    (RayTrainWorker pid=75123) train epoch:[9]      loss:1.727377 [repeated 171x across cluster]
    (RayTrainWorker pid=75123) train epoch:[9]      loss:1.694001 [repeated 182x across cluster]
    
    Training completed after 0 iterations at 2025-03-03 03:41:04. Total running time: 3min 53s
    
    Training result: Result(
      metrics={},
      path='/root/ray_results/TorchTrainer_2025-03-03_03-37-11/TorchTrainer_ca6cf_00000_0_2025-03-03_03-37-11',
      filesystem='local',
      checkpoint=None
    )
    (RayTrainWorker pid=75122) Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    (RayTrainWorker pid=75122) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    (RayTrainWorker pid=75122) train epoch:[9]      loss:0.417845 [repeated 136x across cluster]