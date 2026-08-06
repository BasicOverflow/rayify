# ResNet Model Training with Intel Gaudi

---

# ResNet Model Training with Intel Gaudi

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=intel_gaudi-resnet)   

In this Jupyter notebook, we will train a ResNet-50 model to classify images of ants and bees using HPU. We will use PyTorch for model training and Ray for distributed training. The dataset will be downloaded and processed using torchvision’s datasets and transforms.

[Intel Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Intel Habana Labs. For more information, see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/).

## Configuration

A node with Gaudi/Gaudi2 installed is required to run this example. Both Gaudi and Gaudi2 have 8 HPUs. We will use 2 workers to train the model, each using 1 HPU.

We recommend using a prebuilt container to run these examples. To run a container, you need Docker. See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

Next, follow [Run Using Containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install the Gaudi drivers and container runtime.

Next, start the Gaudi container:
    
    
    docker pull vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    

Inside the container, install Ray and Jupyter to run this notebook.
    
    
    pip install ray[train] notebook
    
    
    
    import os
    from typing import Dict
    from tempfile import TemporaryDirectory
    
    import torch
    from filelock import FileLock
    from torch import nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from tqdm import tqdm
    
    import ray
    import ray.train as train
    from ray.train import ScalingConfig, Checkpoint
    from ray.train.torch import TorchTrainer
    from ray.train.torch import TorchConfig
    from ray.runtime_env import RuntimeEnv
    
    import habana_frameworks.torch.core as htcore
    

## Define Data Transforms

We will set up the data transforms for preprocessing images for training and validation. This includes random cropping, flipping, and normalization for the training set, and resizing and normalization for the validation set.
    
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    

## Dataset Download Function

We will define a function to download the Hymenoptera dataset. This dataset contains images of ants and bees for a binary classification problem.
    
    
    def download_datasets():
        os.system("wget https://download.pytorch.org/tutorial/hymenoptera_data.zip >/dev/null 2>&1")
        os.system("unzip hymenoptera_data.zip >/dev/null 2>&1")
    

## Dataset Preparation Function

After downloading the dataset, we need to build PyTorch datasets for training and validation. The `build_datasets` function will apply the previously defined transforms and create the datasets.
    
    
    def build_datasets():
        torch_datasets = {}
        for split in ["train", "val"]:
            torch_datasets[split] = datasets.ImageFolder(
                os.path.join("./hymenoptera_data", split), data_transforms[split]
            )
        return torch_datasets
    

## Model Initialization Functions

We will define two functions to initialize our model. The `initialize_model` function will load a pre-trained ResNet-50 model and replace the final classification layer for our binary classification task. The `initialize_model_from_checkpoint` function will load a model from a saved checkpoint if available.
    
    
    def initialize_model():
        # Load pretrained model params
        model = models.resnet50(pretrained=True)
    
        # Replace the original classifier with a new Linear layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
    
        # Ensure all params get updated during finetuning
        for param in model.parameters():
            param.requires_grad = True
        return model
    

## Evaluation Function

To assess the performance of our model during training, we define an `evaluate` function. This function computes the number of correct predictions by comparing the predicted labels with the true labels.
    
    
    def evaluate(logits, labels):
        _, preds = torch.max(logits, 1)
        corrects = torch.sum(preds == labels).item()
        return corrects
    

## Training Loop Function

This function defines the training loop that will be executed by each worker. It includes downloading the dataset, preparing data loaders, initializing the model, and running the training and validation phases. Compared to a training function for GPU, no changes are needed to port to HPU. Internally, Ray Train does these things:

  * Detect HPU and set the device.

  * Initializes the habana PyTorch backend.

  * Initializes the habana distributed backend.

    
    
    def train_loop_per_worker(configs):
        import warnings
    
        warnings.filterwarnings("ignore")
    
        # Calculate the batch size for a single worker
        worker_batch_size = configs["batch_size"] // train.get_context().get_world_size()
    
        # Download dataset once on local rank 0 worker
        if train.get_context().get_local_rank() == 0:
            download_datasets()
        torch.distributed.barrier()
    
        # Build datasets on each worker
        torch_datasets = build_datasets()
    
        # Prepare dataloader for each worker
        dataloaders = dict()
        dataloaders["train"] = DataLoader(
            torch_datasets["train"], batch_size=worker_batch_size, shuffle=True
        )
        dataloaders["val"] = DataLoader(
            torch_datasets["val"], batch_size=worker_batch_size, shuffle=False
        )
    
        # Distribute
        dataloaders["train"] = train.torch.prepare_data_loader(dataloaders["train"])
        dataloaders["val"] = train.torch.prepare_data_loader(dataloaders["val"])
    
        # Obtain HPU device automatically
        device = train.torch.get_device()
    
        # Prepare DDP Model, optimizer, and loss function
        model = initialize_model()
        model = model.to(device)
    
        optimizer = optim.SGD(
            model.parameters(), lr=configs["lr"], momentum=configs["momentum"]
        )
        criterion = nn.CrossEntropyLoss()
    
        # Start training loops
        for epoch in range(configs["num_epochs"]):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
    
                    # calculate statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += evaluate(outputs, labels)
    
                size = len(torch_datasets[phase]) // train.get_context().get_world_size()
                epoch_loss = running_loss / size
                epoch_acc = running_corrects / size
    
                if train.get_context().get_world_rank() == 0:
                    print(
                        "Epoch {}-{} Loss: {:.4f} Acc: {:.4f}".format(
                            epoch, phase, epoch_loss, epoch_acc
                        )
                    )
    
                # Report metrics and checkpoint every epoch
                if phase == "val":
                    train.report(
                        metrics={"loss": epoch_loss, "acc": epoch_acc},
                    )
    

## Main Training Function

The `train_resnet` function sets up the distributed training environment using Ray and starts the training process. It specifies the batch size, number of epochs, learning rate, and momentum for the SGD optimizer. To enable training using HPU, we only need to make the following changes:

  * Require an HPU for each worker in ScalingConfig

  * Set backend to “hccl” in TorchConfig

    
    
    def train_resnet(num_workers=2):
        global_batch_size = 16
    
        train_loop_config = {
            "input_size": 224,  # Input image size (224 x 224)
            "batch_size": 32,  # Batch size for training
            "num_epochs": 10,  # Number of epochs to train for
            "lr": 0.001,  # Learning Rate
            "momentum": 0.9,  # SGD optimizer momentum
        }
        # Configure computation resources
        # In ScalingConfig, require an HPU for each worker
        scaling_config = ScalingConfig(num_workers=num_workers, resources_per_worker={"CPU": 1, "HPU": 1})
        # Set backend to hccl in TorchConfig
        torch_config = TorchConfig(backend = "hccl")
        
        ray.init()
        
        # Initialize a Ray TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            torch_config=torch_config,
            scaling_config=scaling_config,
        )
    
        result = trainer.fit()
        print(f"Training result: {result}")
    

## Start Training

Finally, we call the `train_resnet` function to start the training process. You can adjust the number of workers to use. Before running this cell, ensure that Ray is properly set up in your environment to handle distributed training.

Note: the following warning is fine, and is resolved in SynapseAI version 1.14.0+:
    
    
    /usr/local/lib/python3.10/dist-packages/torch/distributed/distributed_c10d.py:252: UserWarning: Device capability of hccl unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.
    
    
    
    train_resnet(num_workers=2) 
    

## Possible outputs
    
    
    2025-03-03 03:32:12,620 INFO worker.py:1841 -- Started a local Ray instance.
    /usr/local/lib/python3.10/dist-packages/ray/tune/impl/tuner_internal.py:125: RayDeprecationWarning: The `RunConfig` class should be imported from `ray.tune` when passing it to the Tuner. Please update your imports. See this issue for more context and migration options: https://github.com/ray-project/ray/issues/49454. Disable these warnings by setting the environment variable: RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
      _log_deprecation_warning(
    (RayTrainWorker pid=63669) Setting up process group for: env:// [rank=0, world_size=2]
    (TorchTrainer pid=63280) Started distributed worker processes: 
    (TorchTrainer pid=63280) - (node_id=9f2c34ea47fe405f3227e9168aa857f81655a83e95fd6be359fd76db, ip=100.83.111.228, pid=63669) world_rank=0, local_rank=0, node_rank=0
    (TorchTrainer pid=63280) - (node_id=9f2c34ea47fe405f3227e9168aa857f81655a83e95fd6be359fd76db, ip=100.83.111.228, pid=63668) world_rank=1, local_rank=1, node_rank=0
    (RayTrainWorker pid=63669) ============================= HABANA PT BRIDGE CONFIGURATION =========================== 
    (RayTrainWorker pid=63669)  PT_HPU_LAZY_MODE = 1
    (RayTrainWorker pid=63669)  PT_HPU_RECIPE_CACHE_CONFIG = ,false,1024
    (RayTrainWorker pid=63669)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
    (RayTrainWorker pid=63669)  PT_HPU_LAZY_ACC_PAR_MODE = 1
    (RayTrainWorker pid=63669)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
    (RayTrainWorker pid=63669)  PT_HPU_EAGER_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=63669)  PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
    (RayTrainWorker pid=63669)  PT_HPU_ENABLE_LAZY_COLLECTIVES = 0
    (RayTrainWorker pid=63669) ---------------------------: System Configuration :---------------------------
    (RayTrainWorker pid=63669) Num CPU Cores : 160
    (RayTrainWorker pid=63669) CPU RAM       : 1056374420 KB
    (RayTrainWorker pid=63669) ------------------------------------------------------------------------------
    (RayTrainWorker pid=63668) Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
      0%|          | 0.00/97.8M [00:00<?, ?B/s]
      9%|▊         | 8.38M/97.8M [00:00<00:01, 87.7MB/s]
    100%|██████████| 97.8M/97.8M [00:00<00:00, 193MB/s]
    100%|██████████| 97.8M/97.8M [00:00<00:00, 203MB/s]
    
    View detailed results here: /root/ray_results/TorchTrainer_2025-03-03_03-32-15
    To visualize your results with TensorBoard, run: `tensorboard --logdir /tmp/ray/session_2025-03-03_03-32-10_695011_53838/artifacts/2025-03-03_03-32-15/TorchTrainer_2025-03-03_03-32-15/driver_artifacts`
    
    Training started with configuration:
    ╭──────────────────────────────────────╮
    │ Training config                      │
    ├──────────────────────────────────────┤
    │ train_loop_config/batch_size      32 │
    │ train_loop_config/input_size     224 │
    │ train_loop_config/lr           0.001 │
    │ train_loop_config/momentum       0.9 │
    │ train_loop_config/num_epochs      10 │
    ╰──────────────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 0-train Loss: 0.6574 Acc: 0.6066
    
    Training finished iteration 1 at 2025-03-03 03:32:45. Total running time: 29s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s       24.684 │
    │ time_total_s           24.684 │
    │ training_iteration          1 │
    │ acc                   0.71053 │
    │ loss                  0.51455 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 0-val Loss: 0.5146 Acc: 0.7105
    (RayTrainWorker pid=63669) Epoch 1-train Loss: 0.5016 Acc: 0.7541
    
    Training finished iteration 2 at 2025-03-03 03:32:46. Total running time: 31s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.39649 │
    │ time_total_s          26.0805 │
    │ training_iteration          2 │
    │ acc                   0.93421 │
    │ loss                  0.30218 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 1-val Loss: 0.3022 Acc: 0.9342
    (RayTrainWorker pid=63669) Epoch 2-train Loss: 0.3130 Acc: 0.9180
    
    Training finished iteration 3 at 2025-03-03 03:32:47. Total running time: 32s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.37042 │
    │ time_total_s          27.4509 │
    │ training_iteration          3 │
    │ acc                   0.93421 │
    │ loss                  0.22201 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 2-val Loss: 0.2220 Acc: 0.9342
    (RayTrainWorker pid=63669) Epoch 3-train Loss: 0.2416 Acc: 0.9262
    
    Training finished iteration 4 at 2025-03-03 03:32:49. Total running time: 34s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.38353 │
    │ time_total_s          28.8345 │
    │ training_iteration          4 │
    │ acc                   0.96053 │
    │ loss                  0.17815 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 3-val Loss: 0.1782 Acc: 0.9605
    (RayTrainWorker pid=63669) Epoch 4-train Loss: 0.1900 Acc: 0.9508
    
    Training finished iteration 5 at 2025-03-03 03:32:50. Total running time: 35s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.37318 │
    │ time_total_s          30.2077 │
    │ training_iteration          5 │
    │ acc                   0.93421 │
    │ loss                  0.17063 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 4-val Loss: 0.1706 Acc: 0.9342
    (RayTrainWorker pid=63669) Epoch 5-train Loss: 0.1346 Acc: 0.9672
    
    Training finished iteration 6 at 2025-03-03 03:32:52. Total running time: 36s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.37999 │
    │ time_total_s          31.5876 │
    │ training_iteration          6 │
    │ acc                   0.96053 │
    │ loss                   0.1552 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 5-val Loss: 0.1552 Acc: 0.9605
    (RayTrainWorker pid=63669) Epoch 6-train Loss: 0.1184 Acc: 0.9672
    
    Training finished iteration 7 at 2025-03-03 03:32:53. Total running time: 38s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.39198 │
    │ time_total_s          32.9796 │
    │ training_iteration          7 │
    │ acc                   0.94737 │
    │ loss                  0.14702 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 6-val Loss: 0.1470 Acc: 0.9474
    (RayTrainWorker pid=63669) Epoch 7-train Loss: 0.0864 Acc: 0.9836
    
    Training finished iteration 8 at 2025-03-03 03:32:54. Total running time: 39s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s       1.3736 │
    │ time_total_s          34.3532 │
    │ training_iteration          8 │
    │ acc                   0.94737 │
    │ loss                  0.14443 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 7-val Loss: 0.1444 Acc: 0.9474
    (RayTrainWorker pid=63669) Epoch 8-train Loss: 0.1085 Acc: 0.9590
    
    Training finished iteration 9 at 2025-03-03 03:32:56. Total running time: 40s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.37868 │
    │ time_total_s          35.7319 │
    │ training_iteration          9 │
    │ acc                   0.94737 │
    │ loss                  0.14194 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 8-val Loss: 0.1419 Acc: 0.9474
    (RayTrainWorker pid=63669) Epoch 9-train Loss: 0.0829 Acc: 0.9754
    
    2025-03-03 03:32:58,628 INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/root/ray_results/TorchTrainer_2025-03-03_03-32-15' in 0.0028s.
    Training finished iteration 10 at 2025-03-03 03:32:57. Total running time: 42s
    ╭───────────────────────────────╮
    │ Training result               │
    ├───────────────────────────────┤
    │ checkpoint_dir_name           │
    │ time_this_iter_s      1.36497 │
    │ time_total_s          37.0969 │
    │ training_iteration         10 │
    │ acc                   0.96053 │
    │ loss                  0.14297 │
    ╰───────────────────────────────╯
    (RayTrainWorker pid=63669) Epoch 9-val Loss: 0.1430 Acc: 0.9605
    
    Training completed after 10 iterations at 2025-03-03 03:32:58. Total running time: 43s
    
    Training result: Result(
      metrics={'loss': 0.1429688463869848, 'acc': 0.9605263157894737},
      path='/root/ray_results/TorchTrainer_2025-03-03_03-32-15/TorchTrainer_19fd8_00000_0_2025-03-03_03-32-15',
      filesystem='local',
      checkpoint=None
    )
    (RayTrainWorker pid=63669) Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
      0%|          | 0.00/97.8M [00:00<?, ?B/s]
     68%|██████▊   | 66.1M/97.8M [00:00<00:00, 160MB/s] [repeated 6x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)