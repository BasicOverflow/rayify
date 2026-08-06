# Fine-tune a Hugging Face Transformers Model

---

# Fine-tune a Hugging Face Transformers Model

This notebook is based on an official Hugging Face example, [How to fine-tune a model on text classification](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). This notebook shows the process of conversion from vanilla HF to Ray Train without changing the training logic unless necessary.

This notebook consists of the following steps:

  1. Set up Ray

  2. Load the dataset

  3. Preprocess the dataset with Ray Data

  4. Run the training with Ray Train

Uncomment and run the following line to install all the necessary dependencies. (This notebook is being tested with `transformers==4.19.1`.):
    
    
    #! pip install "datasets" "transformers>=4.19.0" "torch>=1.10.0" "mlflow"
    

## Set up Ray

Use `ray.init()` to initialize a local cluster. By default, this cluster contains only the machine you are running this notebook on. You can also run this notebook on an [Anyscale](https://www.anyscale.com/) cluster.
    
    
    from pprint import pprint
    import ray
    
    ray.init()
    

Check the resources our cluster is composed of. If you are running this notebook on your local machine or Google Colab, you should see the number of CPU cores and GPUs available on your machine.
    
    
    pprint(ray.cluster_resources())
    
    
    
    {'CPU': 48.0,
     'GPU': 4.0,
     'accelerator_type:T4': 1.0,
     'anyscale/accelerator_shape:4xT4': 1.0,
     'anyscale/node-group:head': 1.0,
     'anyscale/provider:aws': 1.0,
     'anyscale/region:us-west-2': 1.0,
     'memory': 206158430208.0,
     'node:10.0.114.132': 1.0,
     'node:__internal_head__': 1.0,
     'object_store_memory': 58913938636.0}
    

This notebook fine-tunes a [HF Transformers](https://github.com/huggingface/transformers) model for one of the text classification task of the [GLUE Benchmark](https://gluebenchmark.com/). It runs the training using Ray Train.

You can change these two variables to control whether the training, which happens later, uses CPUs or GPUs, and how many workers to spawn. Each worker claims one CPU or GPU. Make sure to not request more resources than the resources present. By default, the training runs with one GPU worker.
    
    
    use_gpu = True  # set this to False to run on CPUs
    num_workers = 1  # set this to number of GPUs or CPUs you want to use
    

## Fine-tune a model on a text classification task

The GLUE Benchmark is a group of nine classification tasks on sentences or pairs of sentences. To learn more, see the [original notebook](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb).

Each task has a name that is its acronym, with `mnli-mm` to indicate that it is a mismatched version of MNLI. Each one has the same training set as `mnli` but different validation and test sets.
    
    
    GLUE_TASKS = [
        "cola",
        "mnli",
        "mnli-mm",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
        "wnli",
    ]
    

This notebook runs on any of the tasks in the list above, with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a version with a classification head. Depending on the model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set these three parameters, and the rest of the notebook should run smoothly:
    
    
    task = "cola"
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16
    

### Loading the dataset

Use the [HF Datasets](https://github.com/huggingface/datasets) library to download the data and get the metric to use for evaluation and to compare your model to the benchmark. You can do this comparison easily with the `load_dataset` and `load_metric` functions.

Apart from `mnli-mm` being special code, you can directly pass the task name to those functions.

Run the normal HF Datasets code to load the dataset from the Hub.
    
    
    from datasets import load_dataset
    
    actual_task = "mnli" if task == "mnli-mm" else task
    datasets = load_dataset("glue", actual_task)
    

The `dataset` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation, and test set, with more keys for the mismatched validation and test set in the special case of `mnli`.

### Preprocessing the data with Ray Data

Before you can feed these texts to the model, you need to preprocess them. Preprocess them with a HF Transformers’ `Tokenizer`, which tokenizes the inputs, including converting the tokens to their corresponding IDs in the pretrained vocabulary, and puts them in a format the model expects. It also generates the other inputs that the model requires.

To do all of this preprocessing, instantiate your tokenizer with the `AutoTokenizer.from_pretrained` method, which ensures that you:

  * Get a tokenizer that corresponds to the model architecture you want to use.

  * Download the vocabulary used when pretraining this specific checkpoint.

    
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    
    
    
    /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    

Pass `use_fast=True` to the preceding call to use one of the fast tokenizers, backed by Rust, from the HF Tokenizers library. These fast tokenizers are available for almost all models, but if you get an error with the previous call, remove the argument.

To preprocess the dataset, you need the names of the columns containing the sentence(s). The following dictionary keeps track of the correspondence task to column names:
    
    
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    

Instead of using HF Dataset objects directly, convert them to [Ray Data](https://docs.ray.io/en/latest/data/data.html). Arrow tables back both of them, so the conversion is straightforward. Use the built-in `from_huggingface()` function.
    
    
    import ray.data
    
    ray_datasets = {
        "train": ray.data.from_huggingface(datasets["train"]),
        "validation": ray.data.from_huggingface(datasets["validation"]),
        "test": ray.data.from_huggingface(datasets["test"]),
    }
    ray_datasets
    
    
    
    {'train': Dataset(num_rows=8551, schema={sentence: string, label: int64, idx: int32}),
     'validation': Dataset(num_rows=1043, schema={sentence: string, label: int64, idx: int32}),
     'test': Dataset(num_rows=1063, schema={sentence: string, label: int64, idx: int32})}
    

You can then write the function that preprocesses the samples. Feed them to the `tokenizer` with the argument `truncation=True`. This configuration ensures that the `tokenizer` truncates and pads to the longest sequence in the batch, any input longer than what the model selected can handle.
    
    
    import numpy as np
    from typing import Dict
    
    
    # Tokenize input sentences
    def collate_fn(examples: Dict[str, np.array]):
        sentence1_key, sentence2_key = task_to_keys[task]
        if sentence2_key is None:
            outputs = tokenizer(
                list(examples[sentence1_key]),
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
        else:
            outputs = tokenizer(
                list(examples[sentence1_key]),
                list(examples[sentence2_key]),
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
    
        outputs["labels"] = torch.LongTensor(examples["label"])
    
        # Move all input tensors to GPU
        for key, value in outputs.items():
            outputs[key] = value.cuda()
    
        return outputs
    

### Fine-tuning the model with Ray Train

Now that the data is ready, download the pretrained model and fine-tune it.

Because all of the tasks involve sentence classification, use the `AutoModelForSequenceClassification` class. For more specifics about each individual training component, see the [original notebook](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). The original notebook uses the same tokenizer used to encode the dataset in this notebook’s preceding example.

The main difference when using Ray Train is that you need to define the training logic as a function (`train_func`). You pass this training function to the `TorchTrainer` to on every Ray worker. The training then proceeds using PyTorch DDP.

Note

Be sure to initialize the model, metric, and tokenizer within the function. Otherwise, you may encounter serialization errors.
    
    
    import torch
    import numpy as np
    
    from evaluate import load
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    
    import ray.train
    from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
    
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation"
        if task == "cola"
        else "accuracy"
    )
    model_name = model_checkpoint.split("/")[-1]
    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )
    name = f"{model_name}-finetuned-{task}"
    
    # Calculate the maximum steps per epoch based on the number of rows in the training dataset.
    # Make sure to scale by the total number of training workers and the per device batch size.
    max_steps_per_epoch = ray_datasets["train"].count() // (batch_size * num_workers)
    
    
    def train_func(config):
        print(f"Is CUDA available: {torch.cuda.is_available()}")
    
        metric = load("glue", actual_task)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
    
        train_ds = ray.train.get_dataset_shard("train")
        eval_ds = ray.train.get_dataset_shard("eval")
    
        train_ds_iterable = train_ds.iter_torch_batches(
            batch_size=batch_size, collate_fn=collate_fn
        )
        eval_ds_iterable = eval_ds.iter_torch_batches(
            batch_size=batch_size, collate_fn=collate_fn
        )
    
        print("max_steps_per_epoch: ", max_steps_per_epoch)
    
        args = TrainingArguments(
            name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=config.get("learning_rate", 2e-5),
            num_train_epochs=config.get("epochs", 2),
            weight_decay=config.get("weight_decay", 0.01),
            push_to_hub=False,
            max_steps=max_steps_per_epoch * config.get("epochs", 2),
            disable_tqdm=True,  # declutter the output a little
            no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
            report_to="none",
        )
    
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if task != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)
    
        trainer = Trainer(
            model,
            args,
            train_dataset=train_ds_iterable,
            eval_dataset=eval_ds_iterable,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    
        trainer.add_callback(RayTrainReportCallback())
    
        trainer = prepare_trainer(trainer)
    
        print("Starting training")
        trainer.train()
    
    
    
    2025-07-09 15:56:28.075767: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-07-09 15:56:28.124864: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2025-07-09 15:56:28.124884: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2025-07-09 15:56:28.126125: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-07-09 15:56:28.133567: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-07-09 15:56:29.219640: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(
    comet_ml is installed but `COMET_API_KEY` is not set.
    

With your `train_func` complete, you can now instantiate the `TorchTrainer`. Aside from calling the function, set the `scaling_config`, which controls the amount of workers and resources used, and the `datasets` to use for training and evaluation.
    
    
    from ray.train.torch import TorchTrainer
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={
            "train": ray_datasets["train"],
            "eval": ray_datasets["validation"],
        },
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    

Finally, call the `fit` method to start training with Ray Train. Save the `Result` object to a variable so you can access metrics and checkpoints.
    
    
    result = trainer.fit()
    
    
    
    2025-07-09 15:56:32,564	INFO tune.py:616 -- [output] This uses the legacy output and progress reporter, as Jupyter notebooks are not supported by the new engine, yet. For more information, please see https://github.com/ray-project/ray/issues/36949
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:32 (running for 00:00:00.11)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 PENDING)
    
    
    
    (TrainTrainable pid=41390) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (TrainTrainable pid=41390)   _torch_pytree._register_pytree_node(
    (TrainTrainable pid=41390) 2025-07-09 15:56:36.371154: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    (TrainTrainable pid=41390) 2025-07-09 15:56:36.418819: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    (TrainTrainable pid=41390) 2025-07-09 15:56:36.418845: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    (TrainTrainable pid=41390) 2025-07-09 15:56:36.420083: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    (TrainTrainable pid=41390) 2025-07-09 15:56:36.427078: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    (TrainTrainable pid=41390) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    (TrainTrainable pid=41390) 2025-07-09 15:56:37.464124: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:37 (running for 00:00:05.13)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 PENDING)
    
    
    
    (TrainTrainable pid=41390) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (TrainTrainable pid=41390)   _torch_pytree._register_pytree_node(
    (TrainTrainable pid=41390) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (TrainTrainable pid=41390)   _torch_pytree._register_pytree_node(
    (TrainTrainable pid=41390) comet_ml is installed but `COMET_API_KEY` is not set.
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:42 (running for 00:00:10.18)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=41521) Setting up process group for: env:// [rank=0, world_size=1]
    (TorchTrainer pid=41390) Started distributed worker processes: 
    (TorchTrainer pid=41390) - (node_id=f67b5f412a227b4c6b3ddd85d6f5b1eecd0bd0917efa8f9cd4b5e4da, ip=10.0.114.132, pid=41521) world_rank=0, local_rank=0, node_rank=0
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (RayTrainWorker pid=41521)   _torch_pytree._register_pytree_node(
    (RayTrainWorker pid=41521) 2025-07-09 15:56:44.730942: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    (RayTrainWorker pid=41521) 2025-07-09 15:56:44.779207: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    (RayTrainWorker pid=41521) 2025-07-09 15:56:44.779230: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    (RayTrainWorker pid=41521) 2025-07-09 15:56:44.780437: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    (RayTrainWorker pid=41521) 2025-07-09 15:56:44.787541: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    (RayTrainWorker pid=41521) To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    (RayTrainWorker pid=41521) 2025-07-09 15:56:45.863740: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (RayTrainWorker pid=41521)   _torch_pytree._register_pytree_node(
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:47 (running for 00:00:15.21)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/accelerator_shape:4xT4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    (RayTrainWorker pid=41521)   _torch_pytree._register_pytree_node(
    (RayTrainWorker pid=41521) comet_ml is installed but `COMET_API_KEY` is not set.
    
    
    
    (RayTrainWorker pid=41521) Is CUDA available: True
    
    
    
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    (RayTrainWorker pid=41521)   warnings.warn(
    (RayTrainWorker pid=41521) Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    (RayTrainWorker pid=41521) You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/ray/data/iterator.py:436: RayDeprecationWarning: Passing a function to `iter_torch_batches(collate_fn)` is deprecated in Ray 2.47. Please switch to using a callable class that inherits from `ArrowBatchCollateFn`, `NumpyBatchCollateFn`, or `PandasBatchCollateFn`.
    (RayTrainWorker pid=41521)   warnings.warn(
    (RayTrainWorker pid=41521) /home/ray/anaconda3/lib/python3.9/site-packages/accelerate/accelerator.py:432: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
    (RayTrainWorker pid=41521) dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)
    (RayTrainWorker pid=41521)   warnings.warn(
    
    
    
    (RayTrainWorker pid=41521) max_steps_per_epoch:  534
    (RayTrainWorker pid=41521) Starting training
    
    
    
    (SplitCoordinator pid=41621) Registered dataset logger for dataset train_23_0
    (SplitCoordinator pid=41621) Starting execution of Dataset train_23_0. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=41621) Execution plan of Dataset train_23_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> OutputSplitter[split(1, equal=True)]
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:52 (running for 00:00:20.23)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (RayTrainWorker pid=41521) /tmp/ipykernel_40967/133795194.py:24: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
    (RayTrainWorker pid=41521) [rank0]:[W reducer.cpp:1389] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
    
    
    
    == Status ==
    Current time: 2025-07-09 15:56:57 (running for 00:00:25.25)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:02 (running for 00:00:30.27)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:07 (running for 00:00:35.29)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:12 (running for 00:00:40.32)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:17 (running for 00:00:45.34)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=41621) ✔️  Dataset train_23_0 execution finished in 28.21 seconds
    
    
    
    (RayTrainWorker pid=41521) {'loss': 0.5441, 'learning_rate': 9.9812734082397e-06, 'epoch': 0.5}
    
    
    
    (SplitCoordinator pid=41622) Registered dataset logger for dataset eval_24_0
    (SplitCoordinator pid=41622) Starting execution of Dataset eval_24_0. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=41622) Execution plan of Dataset eval_24_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> OutputSplitter[split(1, equal=True)]
    
    
    
    == Status ==
    Current time: 2025-07-09 15:57:22 (running for 00:00:50.36)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    (RayTrainWorker pid=41521) {'eval_loss': 0.51453697681427, 'eval_matthews_correlation': 0.37793570732654813, 'eval_runtime': 1.8456, 'eval_samples_per_second': 565.126, 'eval_steps_per_second': 35.761, 'epoch': 0.5}
    
    
    
    2025-07-09 15:57:26,970	WARNING experiment_state.py:206 -- Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds and may become a bottleneck. A snapshot is forced if `CheckpointConfig(num_to_keep)` is set, and a trial has checkpointed >= `num_to_keep` times since the last snapshot.
    You may want to consider increasing the `CheckpointConfig(num_to_keep)` or decreasing the frequency of saving checkpoints.
    You can suppress this warning by setting the environment variable TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S to a smaller value than the current threshold (5.0). Set it to 0 to completely suppress this warning.
    (RayTrainWorker pid=41521) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/TorchTrainer_2025-07-09_15-56-32/TorchTrainer_f5114_00000_0_2025-07-09_15-56-32/checkpoint_000000)
    (SplitCoordinator pid=41622) ✔️  Dataset eval_24_0 execution finished in 1.73 seconds
    
    
    
    == Status ==
    Current time: 2025-07-09 15:57:27 (running for 00:00:55.36)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:32 (running for 00:01:00.38)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:38 (running for 00:01:05.41)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:43 (running for 00:01:10.43)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:48 (running for 00:01:15.45)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/node-group:head, 0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 accelerator_type:T4, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/provider:aws)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    == Status ==
    Current time: 2025-07-09 15:57:53 (running for 00:01:20.47)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    (SplitCoordinator pid=41621) ✔️  Dataset train_23_1 execution finished in 26.58 seconds
    (SplitCoordinator pid=41621) Registered dataset logger for dataset train_23_1
    (SplitCoordinator pid=41621) Starting execution of Dataset train_23_1. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=41621) Execution plan of Dataset train_23_1: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> OutputSplitter[split(1, equal=True)]
    
    
    
    (RayTrainWorker pid=41521) {'loss': 0.3864, 'learning_rate': 0.0, 'epoch': 1.5}
    
    
    
    (SplitCoordinator pid=41622) Registered dataset logger for dataset eval_24_1
    (SplitCoordinator pid=41622) Starting execution of Dataset eval_24_1. Full logs are in /tmp/ray/session_2025-07-09_15-09-59_163606_3385/logs/ray-data
    (SplitCoordinator pid=41622) Execution plan of Dataset eval_24_1: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadParquet] -> OutputSplitter[split(1, equal=True)]
    
    
    
    (RayTrainWorker pid=41521) {'eval_loss': 0.5683005452156067, 'eval_matthews_correlation': 0.45115517656589194, 'eval_runtime': 1.6027, 'eval_samples_per_second': 650.77, 'eval_steps_per_second': 41.18, 'epoch': 1.5}
    == Status ==
    Current time: 2025-07-09 15:57:58 (running for 00:01:25.49)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 RUNNING)
    
    
    
    2025-07-09 15:57:59,354	WARNING experiment_state.py:206 -- Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds and may become a bottleneck. A snapshot is forced if `CheckpointConfig(num_to_keep)` is set, and a trial has checkpointed >= `num_to_keep` times since the last snapshot.
    You may want to consider increasing the `CheckpointConfig(num_to_keep)` or decreasing the frequency of saving checkpoints.
    You can suppress this warning by setting the environment variable TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S to a smaller value than the current threshold (5.0). Set it to 0 to completely suppress this warning.
    (RayTrainWorker pid=41521) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/ray/ray_results/TorchTrainer_2025-07-09_15-56-32/TorchTrainer_f5114_00000_0_2025-07-09_15-56-32/checkpoint_000001)
    (SplitCoordinator pid=41622) ✔️  Dataset eval_24_1 execution finished in 1.49 seconds
    
    
    
    (RayTrainWorker pid=41521) {'train_runtime': 66.7725, 'train_samples_per_second': 255.914, 'train_steps_per_second': 15.995, 'train_loss': 0.4653928092356478, 'epoch': 1.5}
    
    
    
    2025-07-09 15:58:00,649	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/home/ray/ray_results/TorchTrainer_2025-07-09_15-56-32' in 0.0022s.
    2025-07-09 15:58:00,651	INFO tune.py:1041 -- Total run time: 88.09 seconds (88.03 seconds for the tuning loop).
    
    
    
    == Status ==
    Current time: 2025-07-09 15:58:00 (running for 00:01:28.04)
    Using FIFO scheduling algorithm.
    Logical resource usage: 1.0/48 CPUs, 1.0/4 GPUs (0.0/1.0 anyscale/region:us-west-2, 0.0/1.0 anyscale/provider:aws, 0.0/1.0 anyscale/accelerator_shape:4xT4, 0.0/1.0 anyscale/node-group:head, 0.0/1.0 accelerator_type:T4)
    Result logdir: /tmp/ray/session_2025-07-09_15-09-59_163606_3385/artifacts/2025-07-09_15-56-32/TorchTrainer_2025-07-09_15-56-32/driver_artifacts
    Number of trials: 1/1 (1 TERMINATED)
    

You can use the returned `Result` object to access metrics and the Ray Train `Checkpoint` associated with the last iteration.
    
    
    result
    
    
    
    Result(
      metrics={'loss': 0.3864, 'learning_rate': 0.0, 'epoch': 1.5, 'step': 1068, 'eval_loss': 0.5683005452156067, 'eval_matthews_correlation': 0.45115517656589194, 'eval_runtime': 1.6027, 'eval_samples_per_second': 650.77, 'eval_steps_per_second': 41.18},
      path='/home/ray/ray_results/TorchTrainer_2025-07-09_15-56-32/TorchTrainer_f5114_00000_0_2025-07-09_15-56-32',
      filesystem='local',
      checkpoint=Checkpoint(filesystem=local, path=/home/ray/ray_results/TorchTrainer_2025-07-09_15-56-32/TorchTrainer_f5114_00000_0_2025-07-09_15-56-32/checkpoint_000001)
    )
    

## See also

  * [Ray Train Examples](../../examples.md) for more use cases

  * Ray Train User Guides for how-to guides