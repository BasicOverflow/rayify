# Fine-tuning Llama-2 Model with Intel Gaudi

---

# Fine-tuning Llama-2 Model with Intel Gaudi

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=intel_gaudi-llama)   

In this Jupyter notebook, we will:

  * fine-tuning a [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model by using Intel Gaudi accelerators with DDP method

  * fine-tuning a [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) model by using Intel Gaudi accelerators with DeepSpeed method

We will use PyTorch for model training and Ray for distributed training. We will use dataset [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca).

[Intel Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Habana Labs. For more information, see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/).

Basic features for this fine-tuning example are:

  * Running on HPUs, support three execution mode: [“lazy”, “eager”, “eager.compile”](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html).

  * LoRA training.

  * DDP or DeepSpeed based method.

  * [`GaudiTrainer`](https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/transformers/trainer.py) based training.

  * Llama-2-7b/Llama-2-70b model.

  * Ray based resource scheduling and management.

## Prepare environment

This example run on single node with 4 HPUs.

We recommend using a prebuilt container to run these examples. To run a container, you need Docker. See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

Next, follow [Run Using Containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install the Habana drivers and container runtime.

### Get docker image
    
    
    docker pull vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    

### Run docker image
    
    
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    # maybe should mapping your workspace volumns
    

### Install dependency
    
    
    # "optimum-habana>1.11.1" if exection mode "eager" or "eager.compile" 
    # "ray>=2.20.0"
    pip install ray[train] notebook transformers datasets evaluate peft accelerate scikit-learn optimum-habana
    
    # install deepspeed
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.20.0
    
    # this notebook verfied with packages' version:
    # transformers==4.45.2
    # datasets==3.3.2
    # evaluate==0.4.3
    # peft==0.14.0
    # accelerate==0.33.0
    # scikit-learn==1.6.1
    # optimum-habana==1.15.0
    
    # deepspeed==0.16.1+hpu.synapse.v1.20.0
    

## Import necessary libraries
    
    
    import os
    import copy
    from typing import Dict
    
    import torch
    
    import datasets
    import transformers
    from transformers import DataCollatorForLanguageModeling
    
    from tqdm import tqdm
    
    import peft
    
    from optimum.habana import GaudiTrainer, GaudiConfig, GaudiTrainingArguments
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    

## Prepare Dataset Function

Preprocessing the raw dataset’s each line with specified format.
    
    
    def preprocess_dataset(raw_datasets):
    
        PROMPT_DICT = {
            "prompt_with_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_without_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
    
        def create_prompts(examples):
            prompts = {}
            prompts["source"] = []
            prompts["target"] = []
            for example in examples:
                prompt_template = (
                    PROMPT_DICT["prompt_with_input"] if example["input"] != "" else PROMPT_DICT["prompt_without_input"]
                )
                source = prompt_template.format_map(example)
                prompts["source"].append(source)
                prompts["target"].append(example["output"])
            return prompts
    
        # Preprocessing the datasets.
        for key in raw_datasets:
            prompts = create_prompts(raw_datasets[key])
            columns_to_be_removed = list(raw_datasets[key].features.keys())
            raw_datasets[key] = raw_datasets[key].add_column("prompt_sources", prompts["source"])
            raw_datasets[key] = raw_datasets[key].add_column("prompt_targets", prompts["target"])
            raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)
    

## Dataset to Tokenizer Function

Tokenize each line in dataset by model tokenizer.

In example codes, we concatenate the dataset’s line content to accelerate training speed.

All datasets are processed as “train” datasets, no evaluation datasets are sampled from raw_datasets.
    
    
    def preprocess_dataset_to_tokenizer(raw_datasets, tokenizer):
        max_seq_length = 512
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.bos_token_id = 2
    
        def tokenize(prompt, add_eos_token=True):
            results = tokenizer(
                prompt,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_tensors=None,
            )
            for i in range(len(results["input_ids"])):
                if (
                    results["input_ids"][i][-1] != tokenizer.eos_token_id
                    and len(results["input_ids"][i]) < max_seq_length
                    and add_eos_token
                ):
                    results["input_ids"][i].append(tokenizer.eos_token_id)
                    results["attention_mask"][i].append(1)
    
            results["labels"] = copy.deepcopy(results["input_ids"])
            results["input_id_len"] = [len(result) for result in results["input_ids"]]
            return results
    
        def preprocess_function(examples):
            keys = list(examples.data.keys())
            if len(keys) != 2:
                raise ValueError("Unsupported dataset format")
    
            st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]
    
            examples_tokenized = tokenize(st)
            input_ids = examples_tokenized["input_ids"]
            labels = examples_tokenized["labels"]
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": examples_tokenized["attention_mask"],
            }
    
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
        )
    
        def concatenate_data(dataset, max_seq_length):
            concatenated_dataset = {}
            for column in dataset.features:
                concatenated_data = [item for sample in dataset[column] for item in sample]
                reshaped_data = [
                    concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(len(concatenated_data) // max_seq_length)
                ]
                concatenated_dataset[column] = reshaped_data
            return datasets.Dataset.from_dict(concatenated_dataset)
    
        tokenized_datasets_ = tokenized_datasets["train"].remove_columns(["prompt_sources", "prompt_targets"])
        tokenized_datasets["train"] = concatenate_data(tokenized_datasets_, max_seq_length)
    
        return tokenized_datasets
    

## Prepare training arguments

here some arguments are hard coded, you can pass arguments from `config`
    
    
    def prepare_training_args(config: Dict):
        # prepare execution mode config
        execution_mode = config["execution_mode"]
        use_lazy_mode = True if execution_mode == "lazy" else False
        torch_compile_backend = "hpu_backend" if execution_mode == "eager.compile" else None
    
        deepspeed = config["deepspeed"] if "deepspeed" in config else None
    
        return GaudiTrainingArguments(deepspeed=deepspeed,
                                      output_dir=config["output"],
                                      do_train=True,
                                      do_eval=False,
                                      per_device_train_batch_size=config["batch_size_per_worker"],
                                      bf16=True,
                                      learning_rate=config["lr"],
                                      save_strategy="no",
                                      torch_compile_backend=torch_compile_backend,
                                      evaluation_strategy="no",
                                      lr_scheduler_type="cosine",
                                      num_train_epochs=config["epochs"],
                                      use_lazy_mode=use_lazy_mode,
                                      use_habana=True,
                                      pipelining_fwd_bwd=True,
                                      save_only_model=True,
                                      gradient_checkpointing=True,
                                      warmup_ratio=0.03,
                                      throughput_warmup_steps=3,
                                      logging_steps=5)
    

## Prepare model

  1. download model from huggingface or read model from local directory.

  2. convert model to lora model.

  3. move model to HPU device.

If you doesn’t want to fine-tune with LoRA, just remove LoRA conversion step.
    
    
    def prepare_model(config: Dict, device):
        # prepare from pretrained model
        deepspeed = config["deepspeed"] if "deepspeed" in config else None
        if deepspeed is not None:
            auto_config = transformers.AutoConfig.from_pretrained(config["model"], use_cache=False, revision="main", use_auth_token=None, trust_remote_code=None)
            model = transformers.AutoModelForCausalLM.from_pretrained(config["model"], config=auto_config, **config["model_config"])
            model.generation_config.attn_softmax_bf16 = True
            model.generation_config.use_flash_attention = True
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(config["model"], **config["model_config"])
        model.enable_input_require_grads()
    
        # convert to peft model for lora training
        peft_config = peft.LoraConfig(**config["lora_config"])
        model = peft.get_peft_model(model, peft_config)
    
        model.to(dtype=config["model_config"]["torch_dtype"], device=device)
    
        return model
    

## Training Function

This function will be executed by each worker during training, with following steps:

  * preparing training args, an instance of `GaudiTrainingArguments`.

  * loading datasets and preprocess datasets, just load the first 4096 item as training datasets.

  * loading pretrained model as tokenizer, and process datasets to tokenizer.

  * loading pretrained model.

  * preparing data collator and gaidu_config.

  * preparing instance of `GaudiTrainer`.

  * calling `train()` to train model.

  * saving model results.

Compared to a training function for GPU, no changes are needed to port to HPU. Internally, Ray Train does these things:

  * Detect HPU and set the device.

  * Initialize the habana PyTorch backend.

  * Initialize the habana distributed backend.

    
    
    def train_func_per_worker(config: Dict):
        # adapt transformers to gaudi
        adapt_transformers_to_gaudi()
    
        # prepare training arguments
        training_args = prepare_training_args(config)
    
        # prepare datasets
        # here we use dataset "tatsu-lab/alpaca" from huggingface
        raw_datasets = datasets.DatasetDict({"train": datasets.load_dataset("tatsu-lab/alpaca", split='train[0:4096]')})
        preprocess_dataset(raw_datasets)
    
        # prepare tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])
        tokenized_datasets = preprocess_dataset_to_tokenizer(raw_datasets, tokenizer)
    
        # prepare model
        model = prepare_model(config, training_args.device)
    
        # prepare data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False)
    
        # prepare gaudi config
        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True
    
        # instance GaudiTrainer
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
    
        train_result = trainer.train()
        print(f"train_result = {train_result}")
        trainer.save_model()
    
        return train_result
    

## Main Training Function

The `train_llama` function sets up the distributed training environment using Ray and starts the training process. To enable training using HPU, we only need to make the following changes:

  * Set the exectuion mode for training, supported execution mode are:

    * “lazy”: Deferred execution of graphs, comprising of ops delivered from script op by op similar to Eager mode. It gives the Eager mode experience with performance on Gaudi. Unlike Eager Mode with torch.compile, graph is analyzed in each iteration leading to a higher CPU usage.

    * “eager”: Op-by-op execution as defined in standard PyTorch Eager mode scripts.

    * “eager.compile”: Eager mode extended with `torch.compile` \- Similar to Eager mode but extended with wrapping complete or part of model (such as a function) into a graph. Parts that are not wrapped are executed eagerly.

More detail theory can be found [here](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html), and detail performance results can be found [here](https://developer.habana.ai/get-started/habana-models-performance/)

  * Set training method, supported method are:

    * “ddp”

    * “deepspeed”

  * Require an HPU for each worker in ScalingConfig

  * Set backend to `hccl` in TorchConfig

    
    
    def train_llama(num_workers, execution_mode, training_method):
        import ray
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer, TorchConfig
    
        # deepspeed config, can also place it to config file
        deepspeed_config = {
            "steps_per_print": 64,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "bf16": {
                "enabled": True
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": False,
                "contiguous_gradients": False,
                "stage3_gather_16bit_weights_on_model_save": True
            }
        }
    
        # Preparing train configurations
        train_config = {
            "execution_mode": execution_mode,
            "model": "/root/models/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080/",
            "model_config": {"torch_dtype": torch.bfloat16, "trust_remote_code": False, "use_auth_token": None},
            "lora_config": {"task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["q_proj", "v_proj"]},
            "lr": 1e-4,
            "epochs": 2,
            "batch_size_per_worker": 8,
            "output": "/tmp/ray/",
            "deepspeed": deepspeed_config if training_method == "deepspeed" else None,
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

Finally, we call the `train_llama` function to start the training process. You can adjust the number of workers to use, and the execution mode for HPU.
    
    
    # set some environment variables
    os.environ["RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES"] = "0"
    # if using RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES env var
    # you must set HABANA_VISIBLE_DEVICES, such as
    # os.environ["HABANA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # execution_mode are ["lazy", "eager", "eager.compile"]
    execution_mode = "lazy"
    os.environ["PT_HPU_LAZY_MODE"] = "1" if execution_mode == "lazy" else "0"
    
    # training_method are ["ddp", "deepspeed"]
    training_method = "deepspeed"
    if training_method == "deepspeed":
        os.environ["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "10"
        os.environ["DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED"] = "1"
    
    # here use 4 HPUs
    train_llama(num_workers=4, execution_mode=execution_mode, training_method=training_method)
    

## Final output

### For DDP on HPUs

  * Llama-2-70b-chat-hf

  * 4 HPU

  * LoRA

    
    
    (RayTrainWorker pid=123181) {'loss': 1.8051, 'grad_norm': 0.6015625, 'learning_rate': 9.938441702975689e-05, 'epoch': 0.16, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.6754, 'grad_norm': 0.408203125, 'learning_rate': 9.567727288213005e-05, 'epoch': 0.32, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.568, 'grad_norm': 0.4453125, 'learning_rate': 8.885729807284856e-05, 'epoch': 0.48, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.4934, 'grad_norm': 0.4609375, 'learning_rate': 7.938926261462366e-05, 'epoch': 0.65, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.3965, 'grad_norm': 0.3515625, 'learning_rate': 6.7918397477265e-05, 'epoch': 0.81, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.3461, 'grad_norm': 0.34765625, 'learning_rate': 5.522642316338268e-05, 'epoch': 0.97, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.2924, 'grad_norm': 0.32421875, 'learning_rate': 4.2178276747988446e-05, 'epoch': 1.13, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.2643, 'grad_norm': 0.33203125, 'learning_rate': 2.9663167846209998e-05, 'epoch': 1.29, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.263, 'grad_norm': 0.318359375, 'learning_rate': 1.8533980447508137e-05, 'epoch': 1.45, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.2502, 'grad_norm': 0.275390625, 'learning_rate': 9.549150281252633e-06, 'epoch': 1.61, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.2161, 'grad_norm': 0.2734375, 'learning_rate': 3.3209786751399187e-06, 'epoch': 1.77, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=123181) {'loss': 1.2517, 'grad_norm': 0.294921875, 'learning_rate': 2.7390523158633554e-07, 'epoch': 1.94, 'memory_allocated (GB)': 13.64, 'max_memory_allocated (GB)': 48.92, 'total_memory_available (GB)': 94.62}
    

### For DeepSpeed on HPUs

  * Llama-2-70b-chat-hf

  * 4 HPU

  * LoRA

    
    
    (RayTrainWorker pid=50067) {'loss': 1.662, 'grad_norm': 0.36514782905578613, 'learning_rate': 9.938441702975689e-05, 'epoch': 0.16, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.46, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.6047, 'grad_norm': 0.396455317735672, 'learning_rate': 9.567727288213005e-05, 'epoch': 0.32, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.57, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.4974, 'grad_norm': 0.49250370264053345, 'learning_rate': 8.885729807284856e-05, 'epoch': 0.48, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.57, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.4078, 'grad_norm': 0.49840453267097473, 'learning_rate': 7.938926261462366e-05, 'epoch': 0.65, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.57, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.315, 'grad_norm': 0.3432576656341553, 'learning_rate': 6.7918397477265e-05, 'epoch': 0.81, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.2651, 'grad_norm': 0.32175061106681824, 'learning_rate': 5.522642316338268e-05, 'epoch': 0.97, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.1947, 'grad_norm': 0.3646097481250763, 'learning_rate': 4.2178276747988446e-05, 'epoch': 1.13, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.1534, 'grad_norm': 0.4598522186279297, 'learning_rate': 2.9663167846209998e-05, 'epoch': 1.29, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.1404, 'grad_norm': 0.2677183449268341, 'learning_rate': 1.8533980447508137e-05, 'epoch': 1.45, 'memory_allocated (GB)': 32.75, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.1283, 'grad_norm': 0.32087600231170654, 'learning_rate': 9.549150281252633e-06, 'epoch': 1.61, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.0877, 'grad_norm': 0.28305548429489136, 'learning_rate': 3.3209786751399187e-06, 'epoch': 1.77, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=50067) {'loss': 1.1238, 'grad_norm': 0.25713953375816345, 'learning_rate': 2.7390523158633554e-07, 'epoch': 1.94, 'memory_allocated (GB)': 32.86, 'max_memory_allocated (GB)': 94.59, 'total_memory_available (GB)': 94.62}