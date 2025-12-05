# Llama model pre-training on Intel Gaudi

---

# Llama model pre-training on Intel Gaudi

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=intel_gaudi-llama_pretrain)   

In this Jupyter notebook, we will pre-train a [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b) model by using Intel Gaudi accelerators.

We will use PyTorch for model training and Ray for distributed training.

[Intel Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Habana Labs. For more information, see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/).

Basic features for this pre-training example are:

  * Running on HPUs, support three execution mode: [“lazy”, “eager”, “eager.compile”](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html).

  * Pre-training llama model use configuration [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)

  * [`GaudiTrainer`](https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/transformers/trainer.py) based training.

  * DeepSpeed based pre-training.

  * Ray based resource scheduling and management.

## Prepare environment

This example run on single node with 4 HPUs.

We recommend using a prebuilt container to run these examples. To run a container, you need Docker. See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

Next, follow [Run Using Containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install the Habana drivers and container runtime.

### Get docker image
    
    
    # more available docker image can be found here: https://vault.habana.ai/ui/native/gaudi-docker
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
    
    
    #!/usr/bin/env python
    
    import os
    from typing import Any, Dict
    from torch.utils.data import DataLoader
    
    import transformers
    from itertools import chain
    from datasets import load_dataset
    from transformers import default_data_collator
    from transformers.testing_utils import CaptureLogger
    from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
    from optimum.habana.utils import set_seed
    

## Build datasets

Download and load dataset from huggingface.co
    
    
    def load_datasets(config):
        dataset_name = config["name"] 
        dataset_config_name = config["config_name"]
    
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=None,
            token=None,
            streaming=False,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=None,
                token=None,
                streaming=False,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=None,
                token=None,
                streaming=False,
            )
    
        return raw_datasets
    

## Load tokenizer

Download vocabulary from huggingface.co.
    
    
    def load_tokenizer(config):
        name = config["name"]
        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": "main",
            "token": None,
            "trust_remote_code": False,
        }
        return transformers.AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
    

## Tokenize dataset

tokenize word to token ids.
    
    
    def tokenize_dataset(datasets, tokenizer):
        column_names = list(datasets["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]
    
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output
    
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    
        return tokenized_datasets
    

## Group dataset

This preprocssing will concatenate all texts from our dataset and generate chunks of block_size, and will pre-train model much faster.
    
    
    def group_dataset(config, datasets, tokenizer):
        config_name = config["name"]
        auto_config = transformers.AutoConfig.from_pretrained(config_name)
        max_pos_embeddings = auto_config.max_position_embeddings
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            print(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
    
        lm_datasets = datasets.map(
            group_texts,
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        return lm_datasets
    

## Load model

Download and load pre-configed model from huggingface.co, the detail model configurations in config.json
    
    
    def load_model(config):
        name = config["name"]
        model_config = config.get("config", {})
        auto_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path=name, **model_config
        )
        model = transformers.AutoModelForCausalLM.from_config(auto_config, trust_remote_code=False)
    
        return model
    

## Prepare trainer

Instance Trainer with `model`, `gaudi_config`, `training_args`, `tokenizer`

No evaluation dataset passed, just training.
    
    
    def get_trainer(training_args, datasets, tokenizer, model):
        gaudi_config = GaudiConfig.from_pretrained(
            training_args.gaudi_config_name, revision="main",
        )
    
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        return trainer
    

## Training Function

This function will be executed by each worker during training, with following steps:

  * prepare GaudiTrainingArguments object.

  * load datasets from huggingface.co.

  * load pre-configed tokenizer from huggingface.co.

  * tokenize dataset with loaded model tokenizer.

  * concatenate all texts from our dataset and generate chunks of block_size.

  * instance object of `GaudiTrainer` with training_args, datasets, tokenizer, and model.

  * call `train` of trainer.

  * save model.

    
    
    def pretrain_llama(config: Dict[str, Any]):
    
        training_args = GaudiTrainingArguments(**config["training_args"])
        set_seed(training_args.seed)
    
        raw_datasets = load_datasets(config["datasets"])
    
        tokenizer = load_tokenizer(config["tokenizer"])
    
        tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer)
    
        tokenized_datasets = group_dataset(config["model"], tokenized_datasets, tokenizer)
    
        model = load_model(config["model"])
    
        trainer = get_trainer(training_args, tokenized_datasets, tokenizer, model)
    
        result = trainer.train()
        trainer.save_model()
        print(result)
    

## Main Training Function

The `main` function sets up the distributed training environment using Ray and starts the training process. To enable training using HPU, we only need to make the following changes:

  * Set the exectuion mode for training, supported execution mode are:

    * “lazy”: Deferred execution of graphs, comprising of ops delivered from script op by op similar to Eager mode. It gives the Eager mode experience with performance on Gaudi. Unlike Eager Mode with torch.compile, graph is analyzed in each iteration leading to a higher CPU usage.

    * “eager”: Op-by-op execution as defined in standard PyTorch Eager mode scripts.

    * “eager.compile”: Eager mode extended with `torch.compile` \- Similar to Eager mode but extended with wrapping complete or part of model (such as a function) into a graph. Parts that are not wrapped are executed eagerly.

More detail theory can be found [here](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html), and detail performance results can be found [here](https://developer.habana.ai/get-started/habana-models-performance/)

  * Require an HPU for each worker in ScalingConfig

  * Set backend to `hccl` in TorchConfig

    
    
    def main(num_workers, execution_mode):
        import ray
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer, TorchConfig
    
        pretrain_config = {
            "datasets": {
                "name": "wikitext",
                "config_name": "wikitext-2-raw-v1",
            },
            "tokenizer": {
                "name": "huggyllama/llama-7b",
                "config": {}
            },
            "model": {
                "name": "huggyllama/llama-7b",
                "config": {
                    "torch_dtype": "bfloat16",
                },
            },
            "training_args": {
                "per_device_train_batch_size": 1,
                "do_train": True,
                "save_strategy": "no",
                "output_dir": "/tmp/ray/pretrain-llama-2",
                "logging_steps": 1,
                "gaudi_config_name": "Habana/llama",
                "use_habana": True,
                "throughput_warmup_steps": 3,
                "use_lazy_mode": True,
                "overwrite_output_dir": True,
                "seed": 42,
                "bf16": True,
                "report_to":'tensorboard',
                "deepspeed": {
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
                        "reduce_scatter": False,
                        "contiguous_gradients": False,
                        "stage3_gather_16bit_weights_on_model_save": True
                    }
                },
            },
        }
    
        # if execution mode is eager with compile, must spcified with a compile backend
        if execution_mode == "eager.compile":
            pretrain_config["training_args"].update({"torch_compile_backend": "hpu_backend"})
    
        scaling_config = ScalingConfig(num_workers=num_workers,
                                       use_gpu=False,
                                       resources_per_worker={"CPU": 1, "HPU": 1})
    
        # Set backend to hccl in TorchConfig
        torch_config = TorchConfig(backend="hccl")
    
        ray.init()
    
        # Initialize a Ray TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=pretrain_llama,
            train_loop_config=pretrain_config,
            torch_config=torch_config,
            scaling_config=scaling_config
        )
    
        result = trainer.fit()
        print(result)
    

## Start Training

Finally, we call the `main` function to start the pre-training process.

Before calling `main` function, you must set some environment variables.

  1. The visiable devices. Environment variable `HABANA_VISIBLE_DEVICES` and `HABANA_VISIBLE_MODULES` are used to control the HPU device visiable to application, you must set this two environment variable properly. For more detail usage of `HABANA_VISIBLE_DEVICES`, `HABANA_VISIBLE_MODULES`, please visit [here](https://docs.habana.ai/en/latest/PyTorch/Reference/PT_Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html)

  2. The execution mode. Different execution mode has different runtime performance. The default execution mode is lazy mode.

    
    
    # set some environment variables
    os.environ["RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES"] = "0"
    # if using RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES env var
    # you must set HABANA_VISIBLE_MODULES, such as
    # os.environ["HABANA_VISIBLE_MODULES"] = "0,1,2,3"
    
    # execution_mode are ["lazy", "eager", "eager.compile"]
    execution_mode = "lazy"
    os.environ["PT_HPU_LAZY_MODE"] = "1" if execution_mode == "lazy" else "0"
    
    main(num_workers=4, execution_mode=execution_mode)
    

## Possible outputs
    
    
    ...
    
    RayTrainWorker pid=36561) Setting up process group for: env:// [rank=0, world_size=4]
    (TorchTrainer pid=36054) Started distributed worker processes: 
    (TorchTrainer pid=36054) - (node_id=409da2dba1dc3e5b8e58a2b766a4a19d90e7879c28c2fb13644148b8, ip=100.83.111.228, pid=36561) world_rank=0, local_rank=0, node_rank=0
    (TorchTrainer pid=36054) - (node_id=409da2dba1dc3e5b8e58a2b766a4a19d90e7879c28c2fb13644148b8, ip=100.83.111.228, pid=36562) world_rank=1, local_rank=1, node_rank=0
    (TorchTrainer pid=36054) - (node_id=409da2dba1dc3e5b8e58a2b766a4a19d90e7879c28c2fb13644148b8, ip=100.83.111.228, pid=36563) world_rank=2, local_rank=2, node_rank=0
    (TorchTrainer pid=36054) - (node_id=409da2dba1dc3e5b8e58a2b766a4a19d90e7879c28c2fb13644148b8, ip=100.83.111.228, pid=36564) world_rank=3, local_rank=3, node_rank=0
    
    ...
    
    (RayTrainWorker pid=36561) ============================= HABANA PT BRIDGE CONFIGURATION =========================== 
    (RayTrainWorker pid=36561)  PT_HPU_LAZY_MODE = 1
    (RayTrainWorker pid=36561)  PT_HPU_RECIPE_CACHE_CONFIG = ,false,1024
    (RayTrainWorker pid=36561)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
    (RayTrainWorker pid=36561)  PT_HPU_LAZY_ACC_PAR_MODE = 0
    (RayTrainWorker pid=36561)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
    (RayTrainWorker pid=36561)  PT_HPU_EAGER_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=36561)  PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=36561)  PT_HPU_ENABLE_LAZY_COLLECTIVES = 0
    (RayTrainWorker pid=36561) ---------------------------: System Configuration :---------------------------
    (RayTrainWorker pid=36561) Num CPU Cores : 160
    (RayTrainWorker pid=36561) CPU RAM       : 1056374420 KB
    (RayTrainWorker pid=36561) ------------------------------------------------------------------------------
    
    ...
    
    (RayTrainWorker pid=36561) {'loss': 4.1052, 'grad_norm': 2.225008249282837, 'learning_rate': 8.26086956521739e-06, 'epoch': 2.5, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0472, 'grad_norm': 2.0701019763946533, 'learning_rate': 8.212560386473431e-06, 'epoch': 2.51, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.097, 'grad_norm': 2.119075059890747, 'learning_rate': 8.164251207729469e-06, 'epoch': 2.51, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.7035, 'grad_norm': 2.1802899837493896, 'learning_rate': 8.115942028985508e-06, 'epoch': 2.51, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.242, 'grad_norm': 1.9516953229904175, 'learning_rate': 8.067632850241547e-06, 'epoch': 2.52, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9594, 'grad_norm': 2.0580222606658936, 'learning_rate': 8.019323671497584e-06, 'epoch': 2.52, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.3415, 'grad_norm': 2.192605495452881, 'learning_rate': 7.971014492753623e-06, 'epoch': 2.52, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9739, 'grad_norm': 2.0198025703430176, 'learning_rate': 7.922705314009662e-06, 'epoch': 2.52, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1624, 'grad_norm': 2.0957565307617188, 'learning_rate': 7.874396135265701e-06, 'epoch': 2.53, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9744, 'grad_norm': 2.1159448623657227, 'learning_rate': 7.82608695652174e-06, 'epoch': 2.53, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1127, 'grad_norm': 2.159834623336792, 'learning_rate': 7.777777777777777e-06, 'epoch': 2.53, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0588, 'grad_norm': 2.106534004211426, 'learning_rate': 7.729468599033817e-06, 'epoch': 2.54, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.8734, 'grad_norm': 2.445814371109009, 'learning_rate': 7.681159420289856e-06, 'epoch': 2.54, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0278, 'grad_norm': 2.0376927852630615, 'learning_rate': 7.632850241545895e-06, 'epoch': 2.54, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9643, 'grad_norm': 2.1097891330718994, 'learning_rate': 7.584541062801932e-06, 'epoch': 2.54, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1384, 'grad_norm': 2.157325267791748, 'learning_rate': 7.536231884057972e-06, 'epoch': 2.55, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9982, 'grad_norm': 2.230065107345581, 'learning_rate': 7.48792270531401e-06, 'epoch': 2.55, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0988, 'grad_norm': 2.355875015258789, 'learning_rate': 7.439613526570048e-06, 'epoch': 2.55, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0514, 'grad_norm': 2.1178295612335205, 'learning_rate': 7.391304347826088e-06, 'epoch': 2.56, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.9858, 'grad_norm': 2.089723825454712, 'learning_rate': 7.342995169082126e-06, 'epoch': 2.56, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1548, 'grad_norm': 2.308490753173828, 'learning_rate': 7.294685990338164e-06, 'epoch': 2.56, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0356, 'grad_norm': 1.9994627237319946, 'learning_rate': 7.246376811594203e-06, 'epoch': 2.57, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.7696, 'grad_norm': 1.9719663858413696, 'learning_rate': 7.1980676328502416e-06, 'epoch': 2.57, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0157, 'grad_norm': 2.1598856449127197, 'learning_rate': 7.1497584541062814e-06, 'epoch': 2.57, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0113, 'grad_norm': 1.997869849205017, 'learning_rate': 7.10144927536232e-06, 'epoch': 2.57, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1048, 'grad_norm': 2.099222183227539, 'learning_rate': 7.053140096618358e-06, 'epoch': 2.58, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0048, 'grad_norm': 2.100231885910034, 'learning_rate': 7.004830917874397e-06, 'epoch': 2.58, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.0302, 'grad_norm': 2.18204402923584, 'learning_rate': 6.956521739130435e-06, 'epoch': 2.58, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.7227, 'grad_norm': 2.190962553024292, 'learning_rate': 6.908212560386473e-06, 'epoch': 2.59, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.1111, 'grad_norm': 2.349518060684204, 'learning_rate': 6.859903381642513e-06, 'epoch': 2.59, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.024, 'grad_norm': 2.5497331619262695, 'learning_rate': 6.811594202898551e-06, 'epoch': 2.59, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 3.8844, 'grad_norm': 2.3125178813934326, 'learning_rate': 6.7632850241545894e-06, 'epoch': 2.59, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    (RayTrainWorker pid=36561) {'loss': 4.2208, 'grad_norm': 2.1103923320770264, 'learning_rate': 6.7149758454106285e-06, 'epoch': 2.6, 'memory_allocated (GB)': 28.87, 'max_memory_allocated (GB)': 94.26, 'total_memory_available (GB)': 94.62}
    
    ...