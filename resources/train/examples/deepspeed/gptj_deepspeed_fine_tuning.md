# GPT-J-6B Fine-Tuning with Ray Train and DeepSpeed

---

# GPT-J-6B Fine-Tuning with Ray Train and DeepSpeed

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=gptj_deepspeed_fine_tuning)   

This example showcases how to use Ray Train for **GPT-J fine-tuning**. GPT-J is a GPT-2-like causal language model trained on the Pile dataset. This particular model has 6 billion parameters. For more information, see [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj).

This example uses the Ray Train 🤗 Transformers integration and a pre-trained model from the Hugging Face Hub. Note that this example is adaptable to other similar models.

This is an advanced example that focuses on the performance and distributed computing aspects of Ray Train. For a beginner-friendly introduction to the Ray Train 🤗 Transformers integration, see [Basic Example for HuggingFace Transformers](../transformers/transformers_torch_trainer_basic.md#transformers-torch-trainer-basic-example).

Read Ray Train Key Concepts and Ray Data Integration User Guides before starting this example.

Note

To run this example, make sure your Ray cluster has access to at least one GPU with 16 or more GBs of memory. The required amount of memory depends on the model. This notebook is tested with 16 g4dn.4xlarge instances (including the head node).

This notebook has the following steps:

  1. Set up Ray

  2. Load the dataset

  3. Preprocess the dataset with Ray Data

  4. Run the training with Ray Train

  5. Generate text from prompt

Uncomment and run the following line in order to install all the necessary dependencies (this notebook was tested with `accelerate=0.18.0`, `transformers==4.26.0`, `deepspeed==0.12.3`):
    
    
    ! pip install -q "datasets" "evaluate" "accelerate==0.18.0" "transformers==4.26.0" "torch>=1.12.0" "deepspeed==0.12.3"
    
    
    
    import numpy as np
    import pandas as pd
    import os
    

## Set up Ray

First, let’s set some global variables. We will use 16 workers, each being assigned 1 GPU and 8 CPUs.
    
    
    model_name = "EleutherAI/gpt-j-6B"
    use_gpu = True
    num_workers = 16
    cpus_per_worker = 8
    

We will use `ray.init()` to initialize a local cluster. By default, this cluster will be comprised of only the machine you are running this notebook on. You can also run this notebook on an Anyscale cluster.

We define a [runtime environment](../../../ray-core/handling-dependencies.md#runtime-environments) to ensure that the Ray workers have access to all the necessary packages. You can omit the `runtime_env` argument if you have all of the packages already installed on each node in your cluster.
    
    
    import ray
    
    ray.init(
        runtime_env={
            "pip": [
                "datasets",
                "evaluate",
                # The latest combination accelerate==0.25.0, transformers==4.36.0, deepspeed==0.12.4
                # has issues with DeepSpeed process group initialization,
                # and will result in a batch_size validation problem.
                # TODO(ml-team): get rid of the pins once the issue is fixed.
                "accelerate==0.18.0",
                "transformers==4.26.0",
                "torch>=1.12.0",
                "deepspeed==0.12.3",
            ],
        },
    )
    

Show code cell content Hide code cell content
    
    
    # THIS SHOULD BE HIDDEN IN DOCS AND ONLY RAN IN CI
    # Download the model from our S3 mirror as it's faster
    
    import ray
    import subprocess
    import ray.util.scheduling_strategies
    
    
    def force_on_node(node_id: str, remote_func_or_actor_class):
        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False
        )
        options = {"scheduling_strategy": scheduling_strategy}
        return remote_func_or_actor_class.options(**options)
    
    
    def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
        refs = []
        for node in ray.nodes():
            if node["Alive"] and node["Resources"].get("GPU", None):
                refs.append(
                    force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                        **remote_kwargs
                    )
                )
        return ray.get(refs)
    
    
    @ray.remote(num_gpus=1)
    def download_model():
        from transformers.utils.hub import TRANSFORMERS_CACHE
    
        path = os.path.expanduser(
            os.path.join(TRANSFORMERS_CACHE, "models--EleutherAI--gpt-j-6B")
        )
        subprocess.run(["mkdir", "-p", os.path.join(path, "snapshots", "main")])
        subprocess.run(["mkdir", "-p", os.path.join(path, "refs")])
        if os.path.exists(os.path.join(path, "refs", "main")):
            return
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                "--no-sign-request",
                "s3://large-dl-models-mirror/models--EleutherAI--gpt-j-6B/main/",
                os.path.join(path, "snapshots", "main"),
            ]
        )
        with open(os.path.join(path, "snapshots", "main", "hash"), "r") as f:
            f_hash = f.read().strip()
        with open(os.path.join(path, "refs", "main"), "w") as f:
            f.write(f_hash)
        os.rename(
            os.path.join(path, "snapshots", "main"), os.path.join(path, "snapshots", f_hash)
        )
    
    
    _ = run_on_every_node(download_model)
    

## Loading the dataset

We will be fine-tuning the model on the [`tiny_shakespeare` dataset](https://huggingface.co/datasets/tiny_shakespeare), comprised of 40,000 lines of Shakespeare from a variety of Shakespeare’s plays. The aim will be to make the GPT-J model better at generating text in the style of Shakespeare.
    
    
    from datasets import load_dataset
    
    print("Loading tiny_shakespeare dataset")
    current_dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
    current_dataset
    

We will use [Ray Data](https://docs.ray.io/en/latest/data/data.html) for distributed preprocessing and data ingestion. We can easily convert the dataset obtained from Hugging Face Hub to Ray Data by using `ray.data.from_huggingface()`.
    
    
    import ray.data
    
    ray_datasets = {
        "train": ray.data.from_huggingface(current_dataset["train"]),
        "validation": ray.data.from_huggingface(current_dataset["validation"]),
    }
    
    ray_datasets
    
    
    
    {'train': MaterializedDataset(num_blocks=1, num_rows=1, schema={text: string}),
     'validation': MaterializedDataset(num_blocks=1, num_rows=1, schema={text: string})}
    

Note that the dataset is represented by a single line of large string, and needs some preprocessing. To do this, use the `map_batches()` API to apply transformation functions to batches of data.

The `split_text` function takes the single string and splits it into separate lines, removing empty lines and character names ending with ‘:’ (eg. ‘ROMEO:’). The `tokenize` function takes the lines and tokenizes them using the 🤗 Tokenizer associated with the model, ensuring each entry has the same length (`block_size`) by padding and truncating. This preprocessing is necessary for training.

Note

This preprocessing can be done in other ways. A common pattern is to tokenize first, and then split the obtained tokens into equally-sized blocks.
    
    
    block_size = 512
    
    
    
    from transformers import AutoTokenizer
    
    
    def split_text(batch: pd.DataFrame) -> pd.DataFrame:
        text = list(batch["text"])
        flat_text = "".join(text)
        split_text = [
            x.strip()
            for x in flat_text.split("\n")
            if x.strip() and not x.strip()[-1] == ":"
        ]
        return pd.DataFrame(split_text, columns=["text"])
    
    
    def tokenize(batch: pd.DataFrame) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        ret = tokenizer(
            list(batch["text"]),
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_tensors="np",
        )
        ret["labels"] = ret["input_ids"].copy()
        return dict(ret)
    
    
    processed_datasets = {
        key: (
            ds.map_batches(split_text, batch_format="pandas")
            .map_batches(tokenize, batch_format="pandas")
        )
        for key, ds in ray_datasets.items()
    }
    processed_datasets
    
    
    
    {'train': MapBatches(tokenize)
     +- MapBatches(split_text)
        +- Dataset(num_blocks=1, num_rows=1, schema={text: string}),
     'validation': MapBatches(tokenize)
     +- MapBatches(split_text)
        +- Dataset(num_blocks=1, num_rows=1, schema={text: string})}
    

### Fine-tuning the model with Ray Train

Configure Ray Train’s `TorchTrainer` to perform distributed fine-tuning of the model. Specify a `train_loop_per_worker` function, which defines the training logic to be distributed by Ray using Distributed Data Parallelism, which uses the PyTorch Distributed backend internally. Each worker has its own copy of the model, but operates on different data. At the end of each step, all the workers sync gradients.

Because GPT-J is a relatively large model, it may not be possible to fit it on smaller GPU types (<=16 GB GRAM). To deal with that issue, this example uses [DeepSpeed](https://github.com/microsoft/DeepSpeed), a library to optimize the training process and to offload and partition optimizer and parameter states, reducing GRAM usage. Furthermore, DeepSpeed ZeRO Stage 3 can load large models without running out of memory.

🤗 Transformers and Ray Train’s integrations allow you to easily configure and use DDP and DeepSpeed. All you need to do is specify the DeepSpeed configuration in the [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) object.

Tip

There are many DeepSpeed settings that allow you to trade-off speed for memory usage. The settings used below are tailored to the cluster setup used (16 g4dn.4xlarge nodes) and per device batch size of 16. Some things to keep in mind:

  * If your GPUs support bfloat16, use that instead of float16 mixed precision to get better performance and prevent overflows. Replace `fp16=True` with `bf16=True` in `TrainingArguments`.

  * If you are running out of GRAM: try reducing batch size (defined in the cell below the next one), set `"overlap_comm": False` in DeepSpeed config.

  * If you are running out of RAM, add more nodes to your cluster, use nodes with more RAM, set `"pin_memory": False` in the DeepSpeed config, reduce the batch size, and remove `"offload_param"` from the DeepSpeed config.

For more information on DeepSpeed configuration, refer to [Hugging Face documentation](https://huggingface.co/docs/transformers/main_classes/deepspeed) and [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/).

Additionally, if you prefer a lower-level API, the logic below can be expressed as an [Accelerate training loop](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py) distributed by a Ray Train `TorchTrainer`.

#### Training speed

As this example uses data parallelism, each worker operates on its own shard of the data. The batch size set in `train_ds.iter_torch_batches` is the **per device batch size** (per worker batch size). By changing the number of workers, you can change the **effective batch size** and thus the time needed for training to complete. Calculate the effective batch size as `per device batch size * number of workers * number of gradient accumulation steps`. As you add more workers, the effective batch size rises and thus less time is needed to complete a full epoch. While the speedup is not exactly linear due to extra communication overheads, in many cases it can be close to linear.

The preprocessed dataset has 1348 examples. We have set per device batch size to 16.

  * With 16 g4dn.4xlarge nodes, the effective batch size was 256, which equals to 85 steps per epoch. One epoch took **~2440 seconds** (including initialization time).

  * With 32 g4dn.4xlarge nodes, the effective batch size was 512, which equals to 43 steps per epoch. One epoch took **~1280 seconds** (including initialization time).

    
    
    import evaluate
    import torch
    from transformers import (
        Trainer,
        TrainingArguments,
        GPTJForCausalLM,
        AutoTokenizer,
        default_data_collator,
    )
    from transformers.utils.logging import disable_progress_bar, enable_progress_bar
    
    from ray import train
    from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
    
    
    def train_func(config):
        # Use the actual number of CPUs assigned to this worker by Ray
        runtime_ctx = ray.get_runtime_context()
        assigned_cpus = runtime_ctx.get_assigned_resources().get("CPU", 1)
        os.environ["OMP_NUM_THREADS"] = str(int(assigned_cpus))
        # Enable tf32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
    
        batch_size = config.get("batch_size", 4)
        epochs = config.get("epochs", 2)
        warmup_steps = config.get("warmup_steps", 0)
        learning_rate = config.get("learning_rate", 0.00002)
        weight_decay = config.get("weight_decay", 0.01)
        steps_per_epoch = config.get("steps_per_epoch")
    
        deepspeed = {
            "fp16": {
                "enabled": "auto",
                "initial_scale_power": 8,
                "hysteresis": 4,
                "consecutive_hysteresis": True,
            },
            "bf16": {"enabled": "auto"},
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                },
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "gather_16bit_weights_on_model_save": True,
                "round_robin_gradients": True,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 10,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
        }
    
        print("Preparing training arguments")
        training_args = TrainingArguments(
            "output",
            logging_steps=1,
            save_strategy="steps",
            save_steps=steps_per_epoch,
            max_steps=steps_per_epoch * epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            label_names=["input_ids", "attention_mask"],
            push_to_hub=False,
            report_to="none",
            disable_tqdm=True,  # declutter the output a little
            fp16=True,
            gradient_checkpointing=True,
            deepspeed=deepspeed,
        )
        disable_progress_bar()
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    
        print("Loading model")
    
        model = GPTJForCausalLM.from_pretrained(model_name, use_cache=False)
        model.resize_token_embeddings(len(tokenizer))
    
        print("Model loaded")
    
        enable_progress_bar()
    
        metric = evaluate.load("accuracy")
    
        train_ds = train.get_dataset_shard("train")
        eval_ds = train.get_dataset_shard("validation")
    
        train_ds_iterable = train_ds.iter_torch_batches(
            batch_size=batch_size,
            local_shuffle_buffer_size=train.get_context().get_world_size() * batch_size,
        )
        eval_ds_iterable = eval_ds.iter_torch_batches(batch_size=batch_size)
    
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_iterable,
            eval_dataset=eval_ds_iterable,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    
        # Add callback to report checkpoints to Ray Train
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        trainer.train()
    

After defining the training function, instantiate the `TorchTrainer`. Aside from the function, set the `scaling_config` to control the number of workers and amount of resources to use, and `datasets`(the preprocessed Ray Datasets) to use for training and evaluation.

Note

Running with multiple nodes necessitates the persistence of checkpoints and other outputs to some external storage for access after training has completed. **You should set up cloud storage or NFS, then replace` storage_path` with your own cloud bucket URI or NFS path.**

See Configuration and Persistent Storage for more details.
    
    
    storage_path = "s3://your-bucket-here"  # TODO: Set up cloud storage
    # storage_path="/mnt/path/to/nfs"     # TODO: Alternatively, set up NFS
    
    
    
    batch_size = 16
    train_ds_size = processed_datasets["train"].count()
    steps_per_epoch = train_ds_size // (batch_size * num_workers)
    
    
    
    from ray.train.torch import TorchTrainer
    from ray.train import RunConfig, ScalingConfig
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": 1,
            "batch_size": batch_size,  # per device
            "steps_per_epoch": steps_per_epoch,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
        ),
        datasets=processed_datasets,
        run_config=RunConfig(storage_path=storage_path),
    )
    

Finally, call the `fit()` method to start training with Ray Train. Save the `Result` object to a variable to access metrics and checkpoints.
    
    
    results = trainer.fit()
    

Use the returned `Result` object to access metrics and the Ray Train `Checkpoint` associated with the last iteration.
    
    
    checkpoint = results.checkpoint
    checkpoint
    
    
    
    Checkpoint(filesystem=<pyarrow._s3fs.S3FileSystem object at 0x7f8c59d311b0>, path=anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/yunxuan__xiao/gptj-deepspeed-finetune/TorchTrainer_2023-08-18_18-09-11/TorchTrainer_01ea5_00000_0_2023-08-18_18-09-12/checkpoint_000000)
    

### Generate text from prompt

First, download the persistent Ray Train checkpoint from a gpu node and load the fine-tuned model weights and tokenizer from the checkpoint. Then use 🤗 Transformers [`pipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines) to generate predictions from the fine-tuned model.

Tip

For large scale batch inference, see End-to-end: Offline Batch Inference.

Set the `task` to `"text-generation"`, and also set `device_map="auto"` for Ray Train to automatically place the model on the right device.
    
    
    from transformers import pipeline, AutoTokenizer, GPTJForCausalLM
    import os
    
    
    @ray.remote(num_gpus=1)
    def generate_text():
        # Download the checkpoint
        os.system(f"aws s3 sync s3://{checkpoint.path} /mnt/local_storage/")
    
        # Load the model and tokenizer 
        model = GPTJForCausalLM.from_pretrained("/mnt/local_storage/checkpoint")
        tokenizer = AutoTokenizer.from_pretrained("/mnt/local_storage/checkpoint")
    
        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
        # Generate from prompts!
        result = []
        for sentence in pipe(
            ["Romeo and Juliet", "Romeo", "Juliet"], do_sample=True, min_length=20
        ):
            result.append(sentence)
        
        return result
    
    ref = generate_text.remote()
    print(ray.get(ref))
    
    
    
    [{'generated_text': 'Romeo and Juliet. This very night shall they come. A word with you, sir.'}]
    [{'generated_text': 'Romeo! I know thee not. Lord Mercutio, is it you! Signior Montague.'}]
    [{'generated_text': 'Juliet, look up in the vault, and there shalt find a grave; within the monument there is a table:'}]