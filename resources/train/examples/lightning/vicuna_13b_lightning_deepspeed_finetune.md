# Fine-tune vicuna-13b with Lightning and DeepSpeed

---

# Fine-tune `vicuna-13b` with Lightning and DeepSpeed

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=vicuna_13b_lightning_deepspeed_finetune)   

In this example, we will demonstrate how to perform full fine-tuning for a [`vicuna-13b-v1.3`](https://huggingface.co/lmsys/vicuna-13b-v1.3) model using Ray Train PyTorch Lightning integrations with the DeepSpeed ZeRO-3 strategy.

  * [DeepSpeed](https://github.com/microsoft/DeepSpeed) is an open-source deep learning optimization library for PyTorch. It’s designed to reduce computing power and memory usage, and to train large distributed models by leveraging state-of-the-art innovations like ZeRO, 3D-Parallelism, DeepSpeed-MoE, and ZeRO-Infinity.

  * PyTorch Lightning offers a [DeepSpeed integration](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.strategies.DeepSpeedStrategy.html), which provides a simple interface to configure the knobs for DeepSpeed and automatically trigger your training process with the DeepSpeed Engine.

  * `Ray TorchTrainer` allows you to easily scale your PyTorch Lightning job across multiple nodes in a Ray cluster, without worrying about the underlying cluster management, autoscaling, and distributed process group settings.

Our demo aims to illustrate how these three tools can be combined effectively to finetune the Vicuna-13B model, leveraging the strengths of each to create an efficient and high-performance deep learning solution.

Note

This is an advanced example of Large Language Model fine-tuning with Ray Train. If you’re a beginner or new to the concepts of Ray Train and our Lightning integrations, it would be beneficial to first explore the introductory documentation below to build a foundational understanding.

  * Ray Train Key Concepts

  * Ray Data Quickstart

  * [[Basic] Image Classification with PyTorch Lightning and Ray Train](lightning_mnist_example.html)

  * [[Intermediate] Fine-tuning Lightning Modules with with Ray Data](lightning_cola_advanced.html)

## Cluster Setting

### Compute instances

In this example, we set up a Ray cluster on AWS with the following settings:

| num | instance type | GPU per node | GPU Memory | CPU Memory  
---|---|---|---|---|---  
Head node | 1 | g5.16xlarge | 1 x A10G | 24 GB | 256 GB  
Worker node | 15 | g5.4xlarge | 1 x A10G | 24 GB | 64 GB  
  
Note

In this example, we used 16 A10G GPUs for model training and tuned the DeepSpeed configurations for this setup. If you have a different cluster setup or GPUs with lower memory capacities, you may need to modify the DeepSpeed configurations and batch size to fit the model into the GPUs.

Tip

We selected a GPU instance with additional CPU memory for the head node to demonstrate single-node offline inference. If you are training only, you can still opt for the g5.4xlarge instance for the head node.

### Cloud Storage

Additionally, since the checkpoint size for this 13B parameter model can be large (~140GB), we choose to store the checkpoints in AWS S3. Thanks to the newly introduced distributed checkpointing feature in Ray 2.5, each worker can upload its own shards individually to the S3 bucket, greatly reducing the latency and network traffic of checkpoint syncing.

### Local Storage

To demonstrate offline inference, we need to download and consolidate the model checkpoint onto the head node. This action requires around 200GB disk storage. Therefore, we mounted the NVMe SSD provided by g5 instances at `/dev/nvme1n1` to `/mnt/local_storage`, and we will save the checkpoints in this folder.

For more details, see [Amazon EBS and NVMe on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/nvme-ebs-volumes.html).

## Setup Ray Environment

We define a runtime environment to ensure that the Ray workers have access to all necessary packages. If you have already included these dependencies in your Docker image or installed them on each node, you can ignore the `runtime_env` argument.

Note

Note that the codebases of `transformers`, `accelerate`, and `deepspeed` are all rapidly changing, so we have pinned the package versions here to ensure testing stability. You can try other version combinations and feel free to report any issues you encounter.
    
    
    import os
    os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
    
    
    
    import ray
    
    NUM_WORKERS = 16
    BATCH_SIZE_PER_WORKER = 8
    MODEL_NAME = "lmsys/vicuna-13b-v1.3"
    
    ray.init(
        runtime_env={
            "pip": [
                "datasets",
                "torch>=1.13.0",
                "deepspeed==0.12.3",
                "accelerate==0.20.3",
                "transformers==4.30.2",
                "lightning==2.0.3",
            ],
        }
    )
    

## Load and preprocess datasets

We were impressed by LLM’s ability of zero-shot text-generation, while some LLMs may not perform well in code generation due to the lack of code in the training corpus. The CMU [CoNaLa](https://conala-corpus.github.io/)(The Code/Natural Language Challenge) was designed to test systems for generating program snippets from natural language. Each data record contains an intent sentence and a one-line code snippet. The goal is to fine-tune the Vicuna model on this dataset, enabling the model to generate correct and runnable code snippets, thereby achieving natural language intent. Here are some examples:

intent | code snippet  
---|---  
“convert a list of integers into a single integer” | `r = int(''.join(map(str, x)))`  
“normalize a pandas dataframe `df` by row” | `df.div(df.sum(axis=1), axis=0)`  
“Convert string ‘03:55’ into datetime.time object” | `datetime.datetime.strptime('03:55', '%H:%M').time()`  
  
The CoNaLa team has released a dataset crawled from Stack Overflow, automatically filtered, then curated by annotators, split into 2379 training and 500 test examples. In addition, they also included an automatically-mined dataset with 600k examples. In this demo, we take all the curated data and the top 5000 mined data for fine-tuning.

Here we preprocess the CoNaLa dataset with Ray Data. You can also use HuggingFace Datasets and pass it directly to `LightningConfigBuilder.fit_params()`.
    
    
    import re
    import ray
    import json
    from transformers import AutoTokenizer
    from datasets import concatenate_datasets, load_dataset
    
    # Combine the curated dataset and automatically-mined dataset
    hf_dataset_curated = load_dataset("neulab/conala", trust_remote_code=True)
    hf_dataset_mined = load_dataset("neulab/conala", "mined", split="train[:5000]", trust_remote_code=True)
    hf_dataset_merged = concatenate_datasets(
        [hf_dataset_curated["train"], hf_dataset_mined]
    )
    print(hf_dataset_merged)
    
    # Convert it into Ray Dataset
    ray_ds = ray.data.from_huggingface(hf_dataset_merged)
    
    # Build a prompt template for Vicuna-13b model
    PROMPT_TEMPLATE = "Intent: {intent}\nOne-line code snippet: {snippet}"
    
    
    def fill_prompt(batch):
        batch["input_sentence"] = batch.apply(
            lambda row: PROMPT_TEMPLATE.format(
                intent=row["rewritten_intent"]
                if row["rewritten_intent"]
                else row["intent"],
                snippet=f"`{row['snippet']}`",
            )
            + "</s>",
            axis=1,
        )
        return batch[["input_sentence"]]
    
    
    # Tokenize input sentences to tensors
    def tokenize(batch):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side="left", use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        ret = tokenizer(
            list(batch["input_sentence"]),
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="np",
        )
        ret["labels"] = ret["input_ids"].copy()
        return dict(ret)
    
    # Preprocess train dataset
    processed_ds = ray_ds.map_batches(fill_prompt, batch_format="pandas").map_batches(tokenize, batch_format="pandas")
    
    
    
    /home/ray/anaconda3/lib/python3.10/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    
    
    
    Dataset({
        features: ['question_id', 'intent', 'rewritten_intent', 'snippet', 'parent_answer_post_id', 'prob', 'id'],
        num_rows: 7379
    })
    

## Define a Lightning Module

Here we load the pre-trained model weights from HuggingFace Model Hub, and wrap them into `pl.LightningModule`. We adopted the efficient model initialization techniques introduced in [Lightning-transformers](https://github.com/Lightning-Universe/lightning-transformers) to avoid unnecessary full weights loading.
    
    
    import torch
    import transformers
    import lightning.pytorch as pl
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    
    
    class ZeRO3Config:
        def __init__(self, pl_module):
            self.config = pl_module.trainer.strategy.config
    
        def __call__(self, *args, **kwargs):
            return self
    
        def is_zero3(self) -> bool:
            return True
    
    
    def enable_transformers_pretrained_deepspeed_sharding(
        pl_module: "pl.LightningModule",
    ) -> None:
        transformers.deepspeed._hf_deepspeed_config_weak_ref = ZeRO3Config(pl_module)
    
    
    class Vicuna13BModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            # Enable tf32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
    
        def setup(self, stage) -> None:
            # Defer model initialization to inject deepspeed configs to HF.
            # During initialization, HF transformers can immediately partition 
            # the model across all gpus avoid the overhead in time and memory 
            # copying it on CPU or each GPU first.
            enable_transformers_pretrained_deepspeed_sharding(self)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            if self.global_rank == 0:
                print("DeepSpeed Configs: ", self.trainer.strategy.config)
                print("Model Archetecture: ", self.model)
    
        def forward(self, batch):
            outputs = self.model(
                batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            return outputs.loss
    
        def training_step(self, batch, batch_idx):
            loss = self.forward(batch)
            self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
            return loss
    
        def configure_optimizers(self):
            return DeepSpeedCPUAdam(self.parameters(), lr=2e-5, weight_decay=0.01)
    

## DeepSpeed Configurations

Before training, let’s calculate the memory usage of finetuning a `vicuna-13b` model. Assume we are using FP16 mixed-precision training, and the optimizer is Adam with FP32 states.

  * Model parameters: 13(billion parameters) * 2(FP16) ≈ 26GB

  * Optimizer states: 13(billion parameters) * 2(momentums per param) * 4 (FP32) ≈ 52GB

As we can see, the model parameters themselves require 26GB, which cannot fit in a single A10G GPU, let alone the activations and optimizers states. Here, we use ZeRO stage-3 to partition the model, gradients, and optimizer states across 16 nodes. Additionally, we employ optimizer CPU offloading to reduce GRAM usage and increase throughput with larger batch sizes. We also disabled parameter offloading and activation checkpointing to improve the training speed.

Regarding other knobs such as `reduce_bucket_size`, `stage3_prefetch_bucket_size` and `stage3_param_persistence_threshold`, we kept them as the [default values in HuggingFace](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero3-config). Feel free to further adjust them to speed up the training process.
    
    
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    HIDDEN_SIZE = config.hidden_size
    
    deepspeed_configs = {
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": HIDDEN_SIZE * HIDDEN_SIZE,
            "stage3_prefetch_bucket_size": 0.9 * HIDDEN_SIZE * HIDDEN_SIZE,
            "stage3_param_persistence_threshold": 10 * HIDDEN_SIZE,
        },
    }
    
    
    
    /home/ray/anaconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    

## Define your training function

Finally, define the training function that will be launched on multiple workers. The training function is generally the same as the pure pytorch Lightning training code, with additional Ray Train utilities:

  * `RayDeepSpeedStrategy`: Same argument list as Lightning DeepSpeedStrategy but integrated with Ray Train.

  * `RayLightningEnvironment`: Lightning environments for Ray cluster.

  * `RayTrainReportCallback`: On each epoch end, it reports the checkpoint from each worker to the ray train (distributed checkpointing).

  * `prepare_trainer()`: Validate your lightning Trainer configurations.

For Ray Data ingestion, we fetched the preprocessed and sharded dataset with `get_dataset_shard()`, and created a dataloader with `iter_torch_batches()`. It returns a custom iterator that replaces the Torch DataLoader.
    
    
    import ray.train
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.lightning import (
        prepare_trainer,
        RayDeepSpeedStrategy, 
        RayLightningEnvironment, 
        RayTrainReportCallback
    )
    
    
    def train_func(config):
        """Training function for each worker."""
    
        # Unpack the `train_loop_config`
        max_epochs = config["max_epochs"]
        batch_size = config["batch_size"]
        accumulate_grad_batches = config["accumulate_grad_batches"]
    
        model = Vicuna13BModel()
        
        # Prepare Ray Data Ingestion
        train_ds = ray.train.get_dataset_shard("train")
        train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size)
        
        pl_trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            default_root_dir="/mnt/local_storage",
            strategy=RayDeepSpeedStrategy(config=deepspeed_configs),
            plugins=[RayLightningEnvironment()],
            callbacks=[RayTrainReportCallback()],
            enable_checkpointing=False, # RayTrainReportCallback will save the checkpoints
            max_epochs=max_epochs,
            precision="bf16-mixed",
            accumulate_grad_batches=accumulate_grad_batches,
        )
        pl_trainer = prepare_trainer(pl_trainer)
    
        pl_trainer.fit(model, train_dataloaders=train_dataloader)
        
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "max_epochs": 1,
            "batch_size": BATCH_SIZE_PER_WORKER,
            "accumulate_grad_batches": 2,
        },
        run_config=RunConfig(
            name="vicuna-13b-finetune",
            storage_path="/mnt/cluster_storage",
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=True,
            resources_per_worker={"CPU": 15, "GPU": 1},
        ),
        datasets={"train": processed_ds},
    )
    

## Model Fine-tuning

Once everything is configured in TorchTrainer, training becomes easy. Simply call `trainer.fit()`, and your workload will be scaled to the Ray cluster, initiating ZeRO-3 parallel training.
    
    
    result = trainer.fit()
    
    
    
    (TrainController pid=17559) [State Transition] INITIALIZING -> SCHEDULING.
    (TrainController pid=17559) Attempting to start training worker group of size 16 with the following resources: [{'CPU': 15, 'GPU': 1}] * 16
    (RayTrainWorker pid=17770) Setting up process group for: env:// [rank=0, world_size=16]
    
    
    
    (RayTrainWorker pid=17770) [2025-10-15 15:51:07,627] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    
    
    
    (RayTrainWorker pid=4048, ip=10.0.130.188) 2025-10-15 15:51:09.458702: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    (RayTrainWorker pid=4048, ip=10.0.130.188) 2025-10-15 15:51:09.458741: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    (RayTrainWorker pid=4048, ip=10.0.130.188) 2025-10-15 15:51:09.460080: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    (RayTrainWorker pid=4048, ip=10.0.130.188) 2025-10-15 15:51:09.467398: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    (RayTrainWorker pid=4048, ip=10.0.130.188) To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    (RayTrainWorker pid=4048, ip=10.0.130.188) 2025-10-15 15:51:10.359839: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    (RayTrainWorker pid=4048, ip=10.0.130.188) INFO: initializing deepspeed distributed: GLOBAL_RANK: 5, MEMBER: 6/16
    (RayTrainWorker pid=4048, ip=10.0.130.188) initializing deepspeed distributed: GLOBAL_RANK: 5, MEMBER: 6/16
    (RayTrainWorker pid=4048, ip=10.0.130.188) WARNING: Missing logger folder: /tmp/ray/session_2025-10-15_15-40-01_399241_4076/artifacts/vicuna-13b-finetune/lightning_logs
    (RayTrainWorker pid=4048, ip=10.0.130.188) Missing logger folder: /tmp/ray/session_2025-10-15_15-40-01_399241_4076/artifacts/vicuna-13b-finetune/lightning_logs
    (TrainController pid=17559) Started training worker group of size 16: 
    (TrainController pid=17559) - (ip=10.0.171.127, pid=17770) world_rank=0, local_rank=0, node_rank=0
    (TrainController pid=17559) - (ip=10.0.155.201, pid=4224) world_rank=1, local_rank=0, node_rank=1
    (TrainController pid=17559) - (ip=10.0.130.65, pid=4187) world_rank=2, local_rank=0, node_rank=2
    (TrainController pid=17559) - (ip=10.0.178.75, pid=4182) world_rank=3, local_rank=0, node_rank=3
    (TrainController pid=17559) - (ip=10.0.167.159, pid=5417) world_rank=4, local_rank=0, node_rank=4
    (TrainController pid=17559) - (ip=10.0.130.188, pid=4048) world_rank=5, local_rank=0, node_rank=5
    (TrainController pid=17559) - (ip=10.0.134.47, pid=4191) world_rank=6, local_rank=0, node_rank=6
    (TrainController pid=17559) - (ip=10.0.173.126, pid=4079) world_rank=7, local_rank=0, node_rank=7
    (TrainController pid=17559) - (ip=10.0.166.0, pid=4053) world_rank=8, local_rank=0, node_rank=8
    (TrainController pid=17559) - (ip=10.0.183.211, pid=5448) world_rank=9, local_rank=0, node_rank=9
    (TrainController pid=17559) - (ip=10.0.138.121, pid=4069) world_rank=10, local_rank=0, node_rank=10
    (TrainController pid=17559) - (ip=10.0.129.201, pid=5418) world_rank=11, local_rank=0, node_rank=11
    (TrainController pid=17559) - (ip=10.0.184.103, pid=4038) world_rank=12, local_rank=0, node_rank=12
    (TrainController pid=17559) - (ip=10.0.164.99, pid=4075) world_rank=13, local_rank=0, node_rank=13
    (TrainController pid=17559) - (ip=10.0.136.125, pid=4040) world_rank=14, local_rank=0, node_rank=14
    (TrainController pid=17559) - (ip=10.0.161.115, pid=4057) world_rank=15, local_rank=0, node_rank=15
    (TrainController pid=17559) [State Transition] SCHEDULING -> RUNNING.
    (RayTrainWorker pid=17770) INFO: GPU available: True (cuda), used: True
    (RayTrainWorker pid=17770) GPU available: True (cuda), used: True
    (RayTrainWorker pid=17770) INFO: TPU available: False, using: 0 TPU cores
    (RayTrainWorker pid=17770) TPU available: False, using: 0 TPU cores
    (RayTrainWorker pid=17770) INFO: IPU available: False, using: 0 IPUs
    (RayTrainWorker pid=17770) IPU available: False, using: 0 IPUs
    (RayTrainWorker pid=17770) INFO: HPU available: False, using: 0 HPUs
    (RayTrainWorker pid=17770) HPU available: False, using: 0 HPUs
    (RayTrainWorker pid=17770) /home/ray/anaconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    (RayTrainWorker pid=17770)   warnings.warn(
    Downloading shards:   0%|          | 0/3 [00:00<?, ?it/s]
    (RayTrainWorker pid=5418, ip=10.0.129.201) 2025-10-15 15:51:09.590755: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered [repeated 15x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
    (RayTrainWorker pid=5418, ip=10.0.129.201) 2025-10-15 15:51:09.590792: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) 2025-10-15 15:51:09.592129: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) 2025-10-15 15:51:09.599431: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations. [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags. [repeated 15x across cluster]
    
    
    
    (autoscaler +35s) Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
    
    
    
    Downloading shards:  33%|███▎      | 1/3 [00:08<00:16,  8.45s/it]
    (RayTrainWorker pid=5418, ip=10.0.129.201) 2025-10-15 15:51:10.532071: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) INFO: initializing deepspeed distributed: GLOBAL_RANK: 11, MEMBER: 12/16 [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) initializing deepspeed distributed: GLOBAL_RANK: 11, MEMBER: 12/16 [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) WARNING: Missing logger folder: /tmp/ray/session_2025-10-15_15-40-01_399241_4076/artifacts/vicuna-13b-finetune/lightning_logs [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Missing logger folder: /tmp/ray/session_2025-10-15_15-40-01_399241_4076/artifacts/vicuna-13b-finetune/lightning_logs [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) /home/ray/anaconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`. [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201)   warnings.warn( [repeated 15x across cluster]
    Downloading shards:   0%|          | 0/3 [00:00<?, ?it/s] [repeated 15x across cluster]
    Downloading shards:  33%|███▎      | 1/3 [00:14<00:29, 14.64s/it] [repeated 10x across cluster]
    Downloading shards:  33%|███▎      | 1/3 [00:24<00:48, 24.42s/it] [repeated 5x across cluster]
    Downloading shards:  67%|██████▋   | 2/3 [00:32<00:17, 17.90s/it]
    Downloading shards:  67%|██████▋   | 2/3 [00:36<00:19, 19.52s/it]
    Downloading shards:  67%|██████▋   | 2/3 [00:47<00:24, 24.79s/it] [repeated 9x across cluster]
    Downloading shards: 100%|██████████| 3/3 [00:49<00:00, 16.55s/it]
    Downloading shards:  67%|██████▋   | 2/3 [00:51<00:27, 27.69s/it]
    Downloading shards: 100%|██████████| 3/3 [00:54<00:00, 18.33s/it] [repeated 8x across cluster]
    Downloading shards:  67%|██████▋   | 2/3 [01:00<00:33, 33.57s/it]
    Downloading shards: 100%|██████████| 3/3 [00:55<00:00, 18.63s/it]
    Downloading shards: 100%|██████████| 3/3 [01:03<00:00, 21.30s/it]
    Downloading shards:  67%|██████▋   | 2/3 [01:05<00:35, 35.56s/it] [repeated 2x across cluster]
    Downloading shards: 100%|██████████| 3/3 [01:09<00:00, 23.31s/it]
    Downloading shards:  67%|██████▋   | 2/3 [01:12<00:38, 38.09s/it]
    Downloading shards: 100%|██████████| 3/3 [01:30<00:00, 30.22s/it]
    Downloading shards: 100%|██████████| 3/3 [01:36<00:00, 32.00s/it] [repeated 2x across cluster]
    Downloading shards: 100%|██████████| 3/3 [01:41<00:00, 33.94s/it]
    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
    Loading checkpoint shards:  33%|███▎      | 1/3 [00:17<00:35, 17.89s/it]
    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s] [repeated 15x across cluster]
    Loading checkpoint shards:  33%|███▎      | 1/3 [00:23<00:47, 23.70s/it] [repeated 15x across cluster]
    Loading checkpoint shards:  67%|██████▋   | 2/3 [00:39<00:19, 19.88s/it]
    Loading checkpoint shards:  67%|██████▋   | 2/3 [00:39<00:19, 19.89s/it]
    Loading checkpoint shards:  67%|██████▋   | 2/3 [00:44<00:22, 22.24s/it] [repeated 14x across cluster]
    Loading checkpoint shards: 100%|██████████| 3/3 [00:52<00:00, 17.38s/it]
    Loading checkpoint shards: 100%|██████████| 3/3 [00:57<00:00, 19.26s/it] [repeated 15x across cluster]
    (RayTrainWorker pid=4040, ip=10.0.136.125) INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    (RayTrainWorker pid=4040, ip=10.0.136.125) LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    (RayTrainWorker pid=17770) DeepSpeed Configs:  {'zero_allow_untested_optimizer': True, 'bf16': {'enabled': True}, 'zero_optimization': {'stage': 3, 'offload_optimizer': {'device': 'cpu', 'pin_memory': True}, 'overlap_comm': True, 'contiguous_gradients': True, 'reduce_bucket_size': 26214400, 'stage3_prefetch_bucket_size': 23592960.0, 'stage3_param_persistence_threshold': 51200}, 'gradient_accumulation_steps': 2, 'train_micro_batch_size_per_gpu': 1, 'gradient_clipping': 0.0}
    (RayTrainWorker pid=17770) Model Archetecture:  LlamaForCausalLM(
    (RayTrainWorker pid=17770)   (model): LlamaModel(
    (RayTrainWorker pid=17770)     (embed_tokens): Embedding(32000, 5120, padding_idx=0)
    (RayTrainWorker pid=17770)     (layers): ModuleList(
    (RayTrainWorker pid=17770)       (0-39): 40 x LlamaDecoderLayer(
    (RayTrainWorker pid=17770)         (self_attn): LlamaAttention(
    (RayTrainWorker pid=17770)           (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
    (RayTrainWorker pid=17770)           (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
    (RayTrainWorker pid=17770)           (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
    (RayTrainWorker pid=17770)           (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
    (RayTrainWorker pid=17770)           (rotary_emb): LlamaRotaryEmbedding()
    (RayTrainWorker pid=17770)         )
    (RayTrainWorker pid=17770)         (mlp): LlamaMLP(
    (RayTrainWorker pid=17770)           (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
    (RayTrainWorker pid=17770)           (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
    (RayTrainWorker pid=17770)           (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
    (RayTrainWorker pid=17770)           (act_fn): SiLUActivation()
    (RayTrainWorker pid=17770)         )
    (RayTrainWorker pid=17770)         (input_layernorm): LlamaRMSNorm()
    (RayTrainWorker pid=17770)         (post_attention_layernorm): LlamaRMSNorm()
    (RayTrainWorker pid=17770)       )
    (RayTrainWorker pid=17770)     )
    (RayTrainWorker pid=17770)     (norm): LlamaRMSNorm()
    (RayTrainWorker pid=17770)   )
    (RayTrainWorker pid=17770)   (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
    (RayTrainWorker pid=17770) )
    (RayTrainWorker pid=4038, ip=10.0.184.103) Using /home/ray/.cache/torch_extensions/py310_cu121 as PyTorch extensions root...
    (RayTrainWorker pid=4038, ip=10.0.184.103) Creating extension directory /home/ray/.cache/torch_extensions/py310_cu121/cpu_adam...
    (RayTrainWorker pid=4224, ip=10.0.155.201) INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0] [repeated 15x across cluster]
    (RayTrainWorker pid=4224, ip=10.0.155.201) LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0] [repeated 15x across cluster]
    (RayTrainWorker pid=17770) Detected CUDA files, patching ldflags
    (RayTrainWorker pid=17770) Emitting ninja build file /home/ray/.cache/torch_extensions/py310_cu121/cpu_adam/build.ninja...
    (RayTrainWorker pid=17770) /home/ray/anaconda3/lib/python3.10/site-packages/torch/utils/cpp_extension.py:1967: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
    (RayTrainWorker pid=17770) If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
    (RayTrainWorker pid=17770)   warnings.warn(
    (RayTrainWorker pid=17770) Building extension module cpu_adam...
    (RayTrainWorker pid=17770) Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
    
    
    
    (RayTrainWorker pid=4187, ip=10.0.130.65) [1/4] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output custom_cuda_kernel.cuda.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 --use_fast_math -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/common/custom_cuda_kernel.cu -o custom_cuda_kernel.cuda.o 
    (RayTrainWorker pid=5418, ip=10.0.129.201) [2025-10-15 15:51:07,681] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect) [repeated 15x across cluster]
    (RayTrainWorker pid=17770) [2/4] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX256__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o 
    (RayTrainWorker pid=5418, ip=10.0.129.201) [1/4] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output custom_cuda_kernel.cuda.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 --use_fast_math -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/common/custom_cuda_kernel.cu -o custom_cuda_kernel.cuda.o  [repeated 15x across cluster]
    (RayTrainWorker pid=4048, ip=10.0.130.188) [2/4] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX256__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o 
    
    
    
    (RayTrainWorker pid=4048, ip=10.0.130.188) Loading extension module cpu_adam...
    (RayTrainWorker pid=4048, ip=10.0.130.188) Time to load cpu_adam op: 28.735835075378418 seconds
    (RayTrainWorker pid=5418, ip=10.0.129.201) Using /home/ray/.cache/torch_extensions/py310_cu121 as PyTorch extensions root... [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Creating extension directory /home/ray/.cache/torch_extensions/py310_cu121/cpu_adam... [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Detected CUDA files, patching ldflags [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Emitting ninja build file /home/ray/.cache/torch_extensions/py310_cu121/cpu_adam/build.ninja... [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/utils/cpp_extension.py:1967: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.  [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST']. [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201)   warnings.warn( [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Building extension module cpu_adam... [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N) [repeated 15x across cluster]
    
    
    
    (RayTrainWorker pid=4048, ip=10.0.130.188) [4/4] c++ cpu_adam.o cpu_adam_impl.o custom_cuda_kernel.cuda.o -shared -lcurand -L/home/ray/anaconda3/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o cpu_adam.so
    
    
    
    (RayTrainWorker pid=17770) Parameter Offload: Total persistent parameters: 414720 in 81 params
    (RayTrainWorker pid=17770) INFO: 
    (RayTrainWorker pid=17770)   | Name  | Type             | Params | Params per Device
    (RayTrainWorker pid=17770) ---------------------------------------------------------------
    (RayTrainWorker pid=17770) 0 | model | LlamaForCausalLM | 13.0 B | 813 M            
    (RayTrainWorker pid=17770) ---------------------------------------------------------------
    (RayTrainWorker pid=17770) 13.0 B    Trainable params
    (RayTrainWorker pid=17770) 0         Non-trainable params
    (RayTrainWorker pid=17770) 13.0 B    Total params
    (RayTrainWorker pid=17770) 52,063.457Total estimated model params size (MB)
    (RayTrainWorker pid=17770) 
    (RayTrainWorker pid=17770)   | Name  | Type             | Params | Params per Device
    (RayTrainWorker pid=17770) ---------------------------------------------------------------
    (RayTrainWorker pid=17770) 0 | model | LlamaForCausalLM | 13.0 B | 813 M            
    (RayTrainWorker pid=17770) ---------------------------------------------------------------
    (RayTrainWorker pid=17770) 13.0 B    Trainable params
    (RayTrainWorker pid=17770) 0         Non-trainable params
    (RayTrainWorker pid=17770) 13.0 B    Total params
    (RayTrainWorker pid=17770) 52,063.457Total estimated model params size (MB)
    (RayTrainWorker pid=5418, ip=10.0.129.201) Loading extension module cpu_adam... [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) Time to load cpu_adam op: 31.185880184173584 seconds [repeated 15x across cluster]
    
    
    
    Epoch 0: : 0it [00:00, ?it/s]0) 
    (RayTrainWorker pid=5418, ip=10.0.129.201) [2/4] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX256__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o  [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) [3/4] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/TH -isystem /home/ray/anaconda3/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/ray/anaconda3/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX256__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o  [repeated 15x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201) [4/4] c++ cpu_adam.o cpu_adam_impl.o custom_cuda_kernel.cuda.o -shared -lcurand -L/home/ray/anaconda3/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o cpu_adam.so [repeated 15x across cluster]
    
    
    
    (SplitCoordinator pid=17972) Registered dataset logger for dataset train_16_0
    (SplitCoordinator pid=17972) Starting execution of Dataset train_16_0. Full logs are in /tmp/ray/session_2025-10-15_15-40-01_399241_4076/logs/ray-data
    (SplitCoordinator pid=17972) Execution plan of Dataset train_16_0: InputDataBuffer[Input] -> TaskPoolMapOperator[MapBatches(fill_prompt)->MapBatches(tokenize)] -> LimitOperator[limit=2048] -> OutputSplitter[split(16, equal=True)]
    (SplitCoordinator pid=17972) ⚠️  Ray's object store is configured to use only 28.0% of available memory (341.1GiB out of 1216.0GiB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.
    (MapBatches(fill_prompt)->MapBatches(tokenize) pid=4600, ip=10.0.166.0) /home/ray/anaconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    (MapBatches(fill_prompt)->MapBatches(tokenize) pid=4600, ip=10.0.166.0)   warnings.warn(
    (MapBatches(fill_prompt)->MapBatches(tokenize) pid=4600, ip=10.0.166.0) normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
    (SplitCoordinator pid=17972) ✔️  Dataset train_16_0 execution finished in 5.69 seconds
    (RayTrainWorker pid=4048, ip=10.0.130.188) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
    (RayTrainWorker pid=4048, ip=10.0.130.188)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    
    
    
    Epoch 0: : 1it [00:52, 52.00s/it, v_num=0, train_loss=9.190]
    
    
    
    (RayTrainWorker pid=5418, ip=10.0.129.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    (RayTrainWorker pid=4224, ip=10.0.155.201) /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py:1330: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:78.)
    (RayTrainWorker pid=4224, ip=10.0.155.201)   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
    
    
    
    Epoch 0: : 2it [01:28, 44.47s/it, v_num=0, train_loss=9.250]
    
    
    
    (RayTrainWorker pid=4075, ip=10.0.164.99) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4075, ip=10.0.164.99)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    (RayTrainWorker pid=17770) /home/ray/anaconda3/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py:1330: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:78.) [repeated 15x across cluster]
    (RayTrainWorker pid=17770)   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)]) [repeated 15x across cluster]
    
    
    
    Epoch 0: : 3it [01:59, 39.68s/it, v_num=0, train_loss=1.160]
    
    
    
    (RayTrainWorker pid=4040, ip=10.0.136.125) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4040, ip=10.0.136.125)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 4it [02:34, 38.54s/it, v_num=0, train_loss=1.120]
    
    
    
    (RayTrainWorker pid=5418, ip=10.0.129.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=5418, ip=10.0.129.201)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 5it [03:05, 37.12s/it, v_num=0, train_loss=0.957]
    
    
    
    (RayTrainWorker pid=4079, ip=10.0.173.126) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4079, ip=10.0.173.126)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 6it [03:40, 36.73s/it, v_num=0, train_loss=0.941]
    
    
    
    (RayTrainWorker pid=4224, ip=10.0.155.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4224, ip=10.0.155.201)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 7it [04:10, 35.84s/it, v_num=0, train_loss=0.793]
    
    
    
    (RayTrainWorker pid=4182, ip=10.0.178.75) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4182, ip=10.0.178.75)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    (RayTrainWorker pid=4182, ip=10.0.178.75) Exiting prefetcher's background thread
    
    
    
    Epoch 0: : 8it [04:46, 35.78s/it, v_num=0, train_loss=0.777]
    
    
    
    (RayTrainWorker pid=4040, ip=10.0.136.125) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4040, ip=10.0.136.125)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    (RayTrainWorker pid=4079, ip=10.0.173.126) Exiting prefetcher's background thread [repeated 15x across cluster]
    
    
    
    Epoch 0: : 9it [05:19, 35.48s/it, v_num=0, train_loss=0.629]
    
    
    
    (RayTrainWorker pid=4224, ip=10.0.155.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4224, ip=10.0.155.201)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 10it [05:57, 35.70s/it, v_num=0, train_loss=0.672]
    
    
    
    (RayTrainWorker pid=4182, ip=10.0.178.75) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 16x across cluster]
    (RayTrainWorker pid=4182, ip=10.0.178.75)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 16x across cluster]
    
    
    
    Epoch 0: : 11it [06:30, 35.54s/it, v_num=0, train_loss=0.562]
    Epoch 0: : 12it [07:04, 35.41s/it, v_num=0, train_loss=0.562]
    Epoch 0: : 13it [07:36, 35.09s/it, v_num=0, train_loss=0.559]
    Epoch 0: : 14it [08:11, 35.13s/it, v_num=0, train_loss=0.582]
    Epoch 0: : 15it [08:43, 34.89s/it, v_num=0, train_loss=0.535]
    
    
    
    (RayTrainWorker pid=4224, ip=10.0.155.201) /home/ray/anaconda3/lib/python3.10/site-packages/torch/nn/modules/module.py:1898: UserWarning: Positional args are being deprecated, use kwargs instead. Refer to https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict for details.
    (RayTrainWorker pid=4224, ip=10.0.155.201)   warnings.warn(
    (RayTrainWorker pid=4048, ip=10.0.130.188) /home/ray/anaconda3/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.) [repeated 15x across cluster]
    (RayTrainWorker pid=4048, ip=10.0.130.188)   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass [repeated 15x across cluster]
    
    
    
    Epoch 0: : 16it [09:19, 34.98s/it, v_num=0, train_loss=0.551]
    
    
    
    (RayTrainWorker pid=4048, ip=10.0.130.188) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536)
    (RayTrainWorker pid=4048, ip=10.0.130.188) Reporting training result 1: TrainingReport(checkpoint=Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536), metrics={'train_loss': 0.55078125, 'epoch': 0, 'step': 8}, validation_spec=None)
    (RayTrainWorker pid=4075, ip=10.0.164.99) /home/ray/anaconda3/lib/python3.10/site-packages/torch/nn/modules/module.py:1898: UserWarning: Positional args are being deprecated, use kwargs instead. Refer to https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict for details. [repeated 15x across cluster]
    (RayTrainWorker pid=4075, ip=10.0.164.99)   warnings.warn( [repeated 15x across cluster]
    (RayTrainWorker pid=5417, ip=10.0.167.159) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536) [repeated 2x across cluster]
    (RayTrainWorker pid=5417, ip=10.0.167.159) Reporting training result 1: TrainingReport(checkpoint=Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536), metrics={'train_loss': 0.55078125, 'epoch': 0, 'step': 8}, validation_spec=None) [repeated 2x across cluster]
    (RayTrainWorker pid=4191, ip=10.0.134.47) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536) [repeated 8x across cluster]
    (RayTrainWorker pid=4191, ip=10.0.134.47) Reporting training result 1: TrainingReport(checkpoint=Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536), metrics={'train_loss': 0.55078125, 'epoch': 0, 'step': 8}, validation_spec=None) [repeated 8x across cluster]
    (RayTrainWorker pid=4075, ip=10.0.164.99) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536) [repeated 3x across cluster]
    (RayTrainWorker pid=4075, ip=10.0.164.99) Reporting training result 1: TrainingReport(checkpoint=Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536), metrics={'train_loss': 0.55078125, 'epoch': 0, 'step': 8}, validation_spec=None) [repeated 3x across cluster]
    (RayTrainWorker pid=4053, ip=10.0.166.0) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536) [repeated 2x across cluster]
    (RayTrainWorker pid=4053, ip=10.0.166.0) Reporting training result 1: TrainingReport(checkpoint=Checkpoint(filesystem=local, path=/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536), metrics={'train_loss': 0.55078125, 'epoch': 0, 'step': 8}, validation_spec=None) [repeated 2x across cluster]
    (RayTrainWorker pid=17770) INFO: `Trainer.fit` stopped: `max_epochs=1` reached.
    (RayTrainWorker pid=17770) `Trainer.fit` stopped: `max_epochs=1` reached.
    
    
    
    Epoch 0: : 16it [10:25, 39.09s/it, v_num=0, train_loss=0.551]
    
    
    
    (TrainController pid=17559) [State Transition] RUNNING -> FINISHED.
    

## LLM Inference

Now, it’s time to play with our fine-tuned Vicuna code generator!

The deepspeed ZeRO-3 checkpoint is a directory containing of k shards (k=16 in our case).

  * `zero_pp_rank_k_mp_rank_00_model_states.pt`: contains the model parameter skeleton of shard k.

  * `bf16_zero_pp_rank_k_mp_rank_00_optim_states.pt`: contains the actual flattened model parameters and optimizer states of shard k.

Next, we removed the optimizer states and consolidate the checkpoint into a single binary file using DeepSpeed utilities. Also, since we wrapped vicuna-13b within a `LightningModule`, we need to remove the prefix `_forward_module.model.model` so that we can directly load the checkpoint into a HF vicuna model.
    
    
    import os
    import torch
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    def extract_fp32_ckpt_from_zero(zero_ckpt_dir):
        state_dict = get_fp32_state_dict_from_zero_checkpoint(zero_ckpt_dir)
        vicuna_state_dict = {
            k.replace("_forward_module.model.", ""): v for k, v in state_dict.items()
        }
        torch.save(vicuna_state_dict, os.path.join(zero_ckpt_dir, "full_model.pt"))
    
    
    
    Processing zero checkpoint '/mnt/cluster_storage/vicuna-13b-finetune/checkpoint_2025-10-15_16-04-29.037536/checkpoint.ckpt/checkpoint'
    Detected checkpoint of type zero stage 3, world_size: 16
    Parsing checkpoint created by deepspeed==0.12.3
    Reconstructed Trainable fp32 state dict with 363 params 13015864320 elements
    

### Initialize Generation Pipeline

Here, we leverage the Accelerate library to efficiently load the model onto a suitable device(GPU and CPU) and generate a HF text generation pipeline.

  * Initialize an empty model on metadevice

  * Create valid device mappings for the vicuna-13b model

  * Load and distribute model weights to target devices

This ensures that only 1x model size of RAM is used for model initialization.
    
    
    import shutil
    import torch
    import ray
    import lightning.pytorch as pl
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
    from accelerate import (
        init_empty_weights,
        infer_auto_device_map,
        load_checkpoint_and_dispatch,
    )
    
    
    def generate_sample_outputs(model_checkpoint_path, prompts):
        # Initialize a model on meta device
        with init_empty_weights():
            config = AutoConfig.from_pretrained(MODEL_NAME)
            meta_model = AutoModelForCausalLM.from_config(config)
        meta_model.tie_weights()
    
        # Define the device mapping
        device_map = infer_auto_device_map(
            meta_model,
            max_memory={0: "15GB", "cpu": "60GB"},
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    
        local_checkpoint_path = "/mnt/local_storage/vicuna_ckpt"
        shutil.copytree(model_checkpoint_path, local_checkpoint_path)
    
        extract_fp32_ckpt_from_zero(local_checkpoint_path)
    
        full_model_ckpt_path = os.path.join(local_checkpoint_path, "full_model.pt")
    
        # Load the model parameters
        model = load_checkpoint_and_dispatch(
            meta_model,
            checkpoint=full_model_ckpt_path,
            device_map=device_map,
        )
    
        generator = pipeline(
            "text-generation",
            model=model,
            device_map=device_map,
            tokenizer=AutoTokenizer.from_pretrained(
                MODEL_NAME, padding_side="left", use_fast=False
            ),
        )
    
        for sample_prompt in prompts:
            prompt = PROMPT_TEMPLATE.format(intent=sample_prompt["intent"], snippet="")
            output = generator(prompt, max_new_tokens=30, do_sample=True)
            print(output[0]["generated_text"])
    

### Case Study

We took 3 examples from the CoNaLa’s test split for demo:
    
    
    testcases = [
        {
            "intent": "replace white spaces in colunm 'col' of dataframe `df` with '_'",
        },
        {
            "intent": "search for occurrences of regex pattern '>.*<' in xml string `line`",
        },
        {
            "intent": "send a signal `signal.SIGUSR1` to the current process",
        },
    ]
    
    
    
    generate_sample_outputs(os.path.join(result.checkpoint.path, "checkpoint.ckpt"), testcases)
    

### Test the Generated Code Snippets

The generated code snippets look pretty reasonable. The results covered Pandas operations, regular expressions, and Linux commands. Let’s test them one by one.
    
    
    import pandas as pd
    
    df = pd.DataFrame.from_dict({"col": ["abc def ghi", " 12 3 456", "     "]})
    print("Before\n", df)
    
    df["col"] = df["col"].str.replace(" ", "_")
    print("After\n", df)
    
    
    
    import re
    
    line = """
    <bookstore>
      <book category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
      </book>
      <book category="non-fiction">
        <title>Sapiens: A Brief History of Humankind</title>
        <author>Yuval Noah Harari</author>
        <year>2011</year>
      </book>
    </bookstore>
    """
    re.findall(">.*<", line)
    

Finally, let’s hand it over to LLM and let it wrap up the demo:
    
    
    import os, signal
    
    # Don't actually kill the process, it's just for demo :D
    # os.kill(os.getpid(), signal.SIGUSR1)  # Terminate the current process~
    

## References:

  * [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github.io/)

  * [HuggingFace: DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed#deepspeed-integration)

  * [HuggingFace: Handling big models for inference](https://huggingface.co/docs/accelerate/main/usage_guides/big_modeling)

  * [Lightning Transformers: DeepSpeed Training with Big Transformer Models](https://lightning-transformers.readthedocs.io/en/latest/)

  * Rajbhandari, S., Rasley, J., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

  * Zheng, L., Chiang, W-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)