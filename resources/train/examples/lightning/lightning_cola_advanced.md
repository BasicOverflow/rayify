# Fine-tune a PyTorch Lightning Text Classifier with Ray Data

---

# Fine-tune a PyTorch Lightning Text Classifier with Ray Data

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=lightning_cola_advanced)   

Note

This is an intermediate example demonstrates how to use [Ray Data](https://docs.ray.io/en/latest/data/data.html) with PyTorch Lightning in Ray Train.

If you just want to quickly convert your existing PyTorch Lightning scripts into Ray Train, you can refer to the Lightning Quick Start Guide.

This demo introduces how to fine-tune a text classifier on the [CoLA(The Corpus of Linguistic Acceptability)](https://nyu-mll.github.io/CoLA/) dataset using a pre-trained BERT model. In particular, it follows three steps:

  * Preprocess the CoLA dataset with Ray Data.

  * Define a training function with PyTorch Lightning.

  * Launch distributed training with Ray Train’s TorchTrainer.

Run the following line in order to install all the necessary dependencies:
    
    
    !pip install numpy datasets "transformers>=4.19.1" "pytorch_lightning>=1.6.5"
    

Start by importing the needed libraries:
    
    
    import ray
    import torch
    import numpy as np
    import pytorch_lightning as pl
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from evaluate import load
    
    
    
    /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    2025-07-09 16:06:28.571151: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-07-09 16:06:28.619363: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2025-07-09 16:06:28.619382: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2025-07-09 16:06:28.620593: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-07-09 16:06:28.628175: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-07-09 16:06:29.628216: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    

## Pre-process CoLA Dataset

CoLA is a dataset for binary sentence classification with 10.6K training examples. First, download the dataset and metrics using the Hugging Face datasets API, and create a Ray Dataset for each split accordingly.
    
    
    dataset = load_dataset("glue", "cola")
    
    train_dataset = ray.data.from_huggingface(dataset["train"])
    validation_dataset = ray.data.from_huggingface(dataset["validation"])
    

Next, tokenize the input sentences and pad the ID sequence to length 128 using the `bert-base-uncased` tokenizer. The `map_batches` applies this preprocessing function on all data samples.
    
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    def tokenize_sentence(batch):
        outputs = tokenizer(
            batch["sentence"].tolist(),
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        outputs["label"] = batch["label"]
        return outputs
    
    train_dataset = train_dataset.map_batches(tokenize_sentence, batch_format="numpy")
    validation_dataset = validation_dataset.map_batches(tokenize_sentence, batch_format="numpy")
    
    
    
    /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    

## Define a PyTorch Lightning model

You don’t have to make any changes to your `LightningModule` definition. Just copy and paste your code here:
    
    
    class SentimentModel(pl.LightningModule):
        def __init__(self, lr=2e-5, eps=1e-8):
            super().__init__()
            self.lr = lr
            self.eps = eps
            self.num_classes = 2
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-cased", num_labels=self.num_classes
            )
            self.metric = load("glue", "cola")
            self.predictions = []
            self.references = []
    
        def forward(self, batch):
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            return logits
    
        def training_step(self, batch, batch_idx):
            labels = batch["label"]
            logits = self.forward(batch)
            loss = F.cross_entropy(logits.view(-1, self.num_classes), labels)
            self.log("train_loss", loss)
            return loss
    
        def validation_step(self, batch, batch_idx):
            labels = batch["label"]
            logits = self.forward(batch)
            preds = torch.argmax(logits, dim=1)
            self.predictions.append(preds)
            self.references.append(labels)
    
        def on_validation_epoch_end(self):
            predictions = torch.concat(self.predictions).view(-1)
            references = torch.concat(self.references).view(-1)
            matthews_correlation = self.metric.compute(
                predictions=predictions, references=references
            )
    
            # self.metric.compute() returns a dictionary:
            # e.g. {"matthews_correlation": 0.53}
            self.log_dict(matthews_correlation, sync_dist=True)
            self.predictions.clear()
            self.references.clear()
    
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps)
    

## Define a training function

Define a training function that includes all of your lightning training logic. `TorchTrainer` launches this function on each worker in parallel.
    
    
    import ray.train
    from ray.train.lightning import (
        prepare_trainer,
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
    )
    
    train_func_config = {
        "lr": 1e-5,
        "eps": 1e-8,
        "batch_size": 16,
        "max_epochs": 5,
    }
    
    def train_func(config):
        # Unpack the input configs passed from `TorchTrainer(train_loop_config)`
        lr = config["lr"]
        eps = config["eps"]
        batch_size = config["batch_size"]
        max_epochs = config["max_epochs"]
    
        # Fetch the Dataset shards
        train_ds = ray.train.get_dataset_shard("train")
        val_ds = ray.train.get_dataset_shard("validation")
    
        # Create a dataloader for Ray Datasets
        train_ds_loader = train_ds.iter_torch_batches(batch_size=batch_size)
        val_ds_loader = val_ds.iter_torch_batches(batch_size=batch_size)
    
        # Model
        model = SentimentModel(lr=lr, eps=eps)
    
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            callbacks=[RayTrainReportCallback()],
            enable_progress_bar=False,
        )
    
        trainer = prepare_trainer(trainer)
    
        trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=val_ds_loader)
    

To enable distributed training with Ray Train, configure the Lightning Trainer with the following utilities:

  * `RayDDPStrategy`

  * `RayLightningEnvironment`

  * `RayTrainReportCallback`

To ingest Ray Data with Lightning Trainer, follow these three steps:

  * Feed the full Ray dataset to Ray `TorchTrainer` (details in the next section).

  * Use `ray.train.get_dataset_shard` to fetch the sharded dataset on each worker.

  * Use `ds.iter_torch_batches` to create a Ray data loader for Lightning Trainer.

See also

  * Lightning Quick Start Guide

  * User Guides for Ray Data

## Distributed training with Ray TorchTrainer

Next, define a `TorchTrainer` to launch your training function on 4 GPU workers.

You can pass the full Ray dataset to the `datasets` argument of `TorchTrainer`. TorchTrainer automatically shards the datasets among multiple workers.
    
    
    from ray.train.torch import TorchTrainer
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig
    
    
    # Save the top-2 checkpoints according to the evaluation metric
    # The checkpoints and metrics are reported by `RayTrainReportCallback`
    run_config = RunConfig(
        name="ptl-sent-classification",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="matthews_correlation",
            checkpoint_score_order="max",
        ),
    )
    
    # Schedule four workers for DDP training (1 GPU/worker by default)
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_func_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset, "validation": validation_dataset}, # <- Feed the Ray Datasets here
    )
    
    
    
    result = trainer.fit()
    
    
    
    2025-07-09 16:06:43,377	INFO tune.py:616 -- [output] This uses the legacy output and progress reporter, as Jupyter notebooks are not supported by the new engine, yet. For more information, please see https://github.com/ray-project/ray/issues/36949
    
    
    
    == Status ==
    Current time: 2025-07-09 16:06:43 (running for 00:00:00.11)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 PENDING)
    
    
    
    (TrainTrainable pid=47169) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (TrainTrainable pid=47169)   _torch_pytree._register_pytree_node(
    (TrainTrainable pid=47169) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (TrainTrainable pid=47169)   _torch_pytree._register_pytree_node(
    
    
    
    == Status ==
    Current time: 2025-07-09 16:06:48 (running for 00:00:05.13)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 PENDING)
    
    
    
    (TrainTrainable pid=47169) 2025-07-09 16:06:51.068628: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    (TrainTrainable pid=47169) 2025-07-09 16:06:51.116629: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    (TrainTrainable pid=47169) 2025-07-09 16:06:51.116652: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    (TrainTrainable pid=47169) 2025-07-09 16:06:51.117931: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    (TrainTrainable pid=47169) 2025-07-09 16:06:51.125011: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    (TrainTrainable pid=47169) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    (TrainTrainable pid=47169) 2025-07-09 16:06:52.119328: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    
    
    
    == Status ==
    Current time: 2025-07-09 16:06:53 (running for 00:00:10.16)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 PENDING)
    
    
    
    (RayTrainWorker pid=47314) Setting up process group for: env:// [rank=0, world_size=4]
    (TorchTrainer pid=47169) Started distributed worker processes: 
    (TorchTrainer pid=47169) - (node_id=f67b5f412a227b4c6b3ddd85d6f5b1eecd0bd0917efa8f9cd4b5e4da, ip=10.0.114.132, pid=47314) world_rank=0, local_rank=0, node_rank=0
    (TorchTrainer pid=47169) - (node_id=f67b5f412a227b4c6b3ddd85d6f5b1eecd0bd0917efa8f9cd4b5e4da, ip=10.0.114.132, pid=47313) world_rank=1, local_rank=1, node_rank=0
    (TorchTrainer pid=47169) - (node_id=f67b5f412a227b4c6b3ddd85d6f5b1eecd0bd0917efa8f9cd4b5e4da, ip=10.0.114.132, pid=47316) world_rank=2, local_rank=2, node_rank=0
    (TorchTrainer pid=47169) - (node_id=f67b5f412a227b4c6b3ddd85d6f5b1eecd0bd0917efa8f9cd4b5e4da, ip=10.0.114.132, pid=47321) world_rank=3, local_rank=3, node_rank=0
    
    
    
    == Status ==
    Current time: 2025-07-09 16:06:58 (running for 00:00:15.19)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=47314) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (RayTrainWorker pid=47314)   _torch_pytree._register_pytree_node(
    (RayTrainWorker pid=47314) 2025-07-09 16:07:03.237463: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    (RayTrainWorker pid=47314) 2025-07-09 16:07:03.285818: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    (RayTrainWorker pid=47314) 2025-07-09 16:07:03.285846: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    (RayTrainWorker pid=47314) 2025-07-09 16:07:03.287089: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    (RayTrainWorker pid=47314) 2025-07-09 16:07:03.294281: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    (RayTrainWorker pid=47314) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:03 (running for 00:00:20.21)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=47314) 2025-07-09 16:07:04.341505: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    (SplitCoordinator pid=47667) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead. [repeated 9x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
    (SplitCoordinator pid=47667)   _torch_pytree._register_pytree_node( [repeated 9x across cluster]
    (RayTrainWorker pid=47314) /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    (RayTrainWorker pid=47314)   warnings.warn(
    (RayTrainWorker pid=47314) Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    (RayTrainWorker pid=47314) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    (RayTrainWorker pid=47321) Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    (RayTrainWorker pid=47314) /home/ray/anaconda3/lib/python3.9/site-packages/ray/train/lightning/_lightning_utils.py:262: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning"
    (RayTrainWorker pid=47314) `get_trial_name` is deprecated because the concept of a `Trial` will soon be removed in Ray Train.Ray Train will no longer assume that it's running within a Ray Tune `Trial` in the future. See this issue for more context and migration options: https://github.com/ray-project/ray/issues/49454. Disable these warnings by setting the environment variable: RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
    (RayTrainWorker pid=47314)   self.trial_name = train.get_context().get_trial_name()
    (RayTrainWorker pid=47314) GPU available: True (cuda), used: True
    (RayTrainWorker pid=47314) TPU available: False, using: 0 TPU cores
    (RayTrainWorker pid=47314) IPU available: False, using: 0 IPUs
    (RayTrainWorker pid=47314) HPU available: False, using: 0 HPUs
    (RayTrainWorker pid=47314) Missing logger folder: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/working_dirs/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/lightning_logs
    (RayTrainWorker pid=47314) LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
    (RayTrainWorker pid=47314) 
    (RayTrainWorker pid=47314)   | Name  | Type                          | Params
    (RayTrainWorker pid=47314) --------------------------------------------------------
    (RayTrainWorker pid=47314) 0 | model | BertForSequenceClassification | 108 M 
    (RayTrainWorker pid=47314) --------------------------------------------------------
    (RayTrainWorker pid=47314) 108 M     Trainable params
    (RayTrainWorker pid=47314) 0         Non-trainable params
    (RayTrainWorker pid=47314) 108 M     Total params
    (RayTrainWorker pid=47314) 433.247   Total estimated model params size (MB)
    (SplitCoordinator pid=47666) Registered dataset logger for dataset validation_40_0
    (SplitCoordinator pid=47666) Starting execution of Dataset validation_40_0. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=47666) Execution plan of Dataset validation_40_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> TaskPoolMapOperator[MapBatches(tokenize_sentence)->MapBatches(random_sample)] -> OutputSplitter[split(4, equal=True)]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:08 (running for 00:00:25.23)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 16:07:13 (running for 00:00:30.26)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=47321) 2025-07-09 16:07:03.305020: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`. [repeated 3x across cluster]
    (RayTrainWorker pid=47321) 2025-07-09 16:07:03.353280: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered [repeated 3x across cluster]
    (RayTrainWorker pid=47321) 2025-07-09 16:07:03.353303: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered [repeated 3x across cluster]
    (RayTrainWorker pid=47321) 2025-07-09 16:07:03.354507: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered [repeated 3x across cluster]
    (RayTrainWorker pid=47321) 2025-07-09 16:07:03.361526: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations. [repeated 3x across cluster]
    (RayTrainWorker pid=47321) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags. [repeated 3x across cluster]
    (RayTrainWorker pid=47321) 2025-07-09 16:07:04.397838: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT [repeated 3x across cluster]
    (MapBatches(tokenize_sentence)->MapBatches(random_sample) pid=48062) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead. [repeated 5x across cluster]
    (MapBatches(tokenize_sentence)->MapBatches(random_sample) pid=48062)   _torch_pytree._register_pytree_node( [repeated 5x across cluster]
    (RayTrainWorker pid=47321) /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`. [repeated 3x across cluster]
    (RayTrainWorker pid=47321)   warnings.warn( [repeated 3x across cluster]
    (RayTrainWorker pid=47316) Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight'] [repeated 2x across cluster]
    (RayTrainWorker pid=47316) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference. [repeated 3x across cluster]
    (RayTrainWorker pid=47321) /home/ray/anaconda3/lib/python3.9/site-packages/ray/train/lightning/_lightning_utils.py:262: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning" [repeated 3x across cluster]
    (RayTrainWorker pid=47321) `get_trial_name` is deprecated because the concept of a `Trial` will soon be removed in Ray Train.Ray Train will no longer assume that it's running within a Ray Tune `Trial` in the future. See this issue for more context and migration options: https://github.com/ray-project/ray/issues/49454. Disable these warnings by setting the environment variable: RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0 [repeated 3x across cluster]
    (RayTrainWorker pid=47321)   self.trial_name = train.get_context().get_trial_name() [repeated 3x across cluster]
    (RayTrainWorker pid=47313) Missing logger folder: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/working_dirs/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/lightning_logs [repeated 3x across cluster]
    (RayTrainWorker pid=47321) LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3] [repeated 3x across cluster]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:18 (running for 00:00:35.29)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=47666) ✔️  Dataset validation_40_0 execution finished in 11.01 seconds
    
    
    
    (SplitCoordinator pid=47667) Registered dataset logger for dataset train_39_0
    (SplitCoordinator pid=47667) Starting execution of Dataset train_39_0. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=47667) Execution plan of Dataset train_39_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> TaskPoolMapOperator[MapBatches(tokenize_sentence)->MapBatches(random_sample)] -> OutputSplitter[split(4, equal=True)]
    (RayTrainWorker pid=47314) [rank0]:[W reducer.cpp:1389] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
    (MapBatches(tokenize_sentence)->MapBatches(random_sample) pid=50017) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead. [repeated 46x across cluster]
    (MapBatches(tokenize_sentence)->MapBatches(random_sample) pid=50017)   _torch_pytree._register_pytree_node( [repeated 46x across cluster]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:23 (running for 00:00:40.31)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=47667) ✔️  Dataset train_39_0 execution finished in 2.76 seconds
    (SplitCoordinator pid=47666) Registered dataset logger for dataset validation_40_1
    (SplitCoordinator pid=47666) Starting execution of Dataset validation_40_1. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=47666) Execution plan of Dataset validation_40_1: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> TaskPoolMapOperator[MapBatches(tokenize_sentence)->MapBatches(random_sample)] -> OutputSplitter[split(4, equal=True)]
    
    
    
    (SplitCoordinator pid=47666) ✔️  Dataset validation_40_1 execution finished in 2.42 seconds
    (RayTrainWorker pid=47321) [rank3]:[W reducer.cpp:1389] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator()) [repeated 3x across cluster]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:28 (running for 00:00:45.34)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=47316) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/checkpoint_000000)
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:33 (running for 00:00:50.38)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=47667) Registered dataset logger for dataset train_39_1
    (SplitCoordinator pid=47667) Starting execution of Dataset train_39_1. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=47667) Execution plan of Dataset train_39_1: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> TaskPoolMapOperator[MapBatches(tokenize_sentence)->MapBatches(random_sample)] -> OutputSplitter[split(4, equal=True)]
    (RayTrainWorker pid=47314) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/checkpoint_000000) [repeated 3x across cluster]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:38 (running for 00:00:55.41)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=47667) ✔️  Dataset train_39_1 execution finished in 3.11 seconds
    (SplitCoordinator pid=47666) Registered dataset logger for dataset validation_40_2
    (SplitCoordinator pid=47666) Starting execution of Dataset validation_40_2. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=47666) Execution plan of Dataset validation_40_2: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> TaskPoolMapOperator[MapBatches(tokenize_sentence)->MapBatches(random_sample)] -> OutputSplitter[split(4, equal=True)]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:43 (running for 00:01:00.43)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=47666) ✔️  Dataset validation_40_2 execution finished in 2.39 seconds
    (RayTrainWorker pid=47316) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/checkpoint_000001)
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:48 (running for 00:01:05.46)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=47314) `Trainer.fit` stopped: `max_epochs=2` reached.
    (RayTrainWorker pid=47314) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/checkpoint_000001) [repeated 3x across cluster]
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:53 (running for 00:01:10.49)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    2025-07-09 16:07:54,970	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/home/ray/ray_results/ptl-sent-classification' in 0.0022s.
    2025-07-09 16:07:54,972	INFO tune.py:1041 -- Total run time: 71.59 seconds (71.58 seconds for the tuning loop).
    
    
    
    == Status ==
    Current time: 2025-07-09 16:07:54 (running for 00:01:11.59)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_16-06-43/ptl-sent-classification/driver_artifacts
    Number of trials: 1/1 (1 TERMINATED)
    

Note

Note that this examples uses Ray Data for data ingestion for faster preprocessing, but you can also continue to use the native `PyTorch DataLoader` or `LightningDataModule`. See [Train a Pytorch Lightning Image Classifier](lightning_mnist_example.md).
    
    
    result
    
    
    
    Result(
      metrics={'train_loss': 0.8652846813201904, 'matthews_correlation': 0.0, 'epoch': 1, 'step': 28},
      path='/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43',
      filesystem='local',
      checkpoint=Checkpoint(filesystem=local, path=/home/ray/ray_results/ptl-sent-classification/TorchTrainer_61240_00000_0_2025-07-09_16-06-43/checkpoint_000001)
    )
    

## See also

  * [Ray Train Examples](../../examples.md) for more use cases

  * Ray Train User Guides for how-to guides