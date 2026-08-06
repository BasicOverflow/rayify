# Train a Pytorch Lightning Image Classifier

---

# Train a Pytorch Lightning Image Classifier

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=lightning_mnist_example)   

This example introduces how to train a Pytorch Lightning Module using Ray Train `TorchTrainer`. It demonstrates how to train a basic neural network on the MNIST dataset with distributed data parallelism.
    
    
    !pip install "torchmetrics>=0.9" "pytorch_lightning>=1.6" 
    
    
    
    import os
    import numpy as np
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from filelock import FileLock
    from torch.utils.data import DataLoader, random_split, Subset
    from torchmetrics import Accuracy
    from torchvision.datasets import MNIST
    from torchvision import transforms
    
    import pytorch_lightning as pl
    from pytorch_lightning import trainer
    from pytorch_lightning.loggers.csv_logs import CSVLogger
    

## Prepare a dataset and module

The Pytorch Lightning Trainer takes either `torch.utils.data.DataLoader` or `pl.LightningDataModule` as data inputs. You can continue using them without any changes with Ray Train.
    
    
    class MNISTDataModule(pl.LightningDataModule):
        def __init__(self, batch_size=100):
            super().__init__()
            self.data_dir = os.getcwd()
            self.batch_size = batch_size
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
    
        def setup(self, stage=None):
            with FileLock(f"{self.data_dir}.lock"):
                mnist = MNIST(
                    self.data_dir, train=True, download=True, transform=self.transform
                )
    
                # Split data into train and val sets
                self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
    
        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)
    
        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)
    
        def test_dataloader(self):
            with FileLock(f"{self.data_dir}.lock"):
                self.mnist_test = MNIST(
                    self.data_dir, train=False, download=True, transform=self.transform
                )
            return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
    

Next, define a simple multi-layer perception as the subclass of `pl.LightningModule`.
    
    
    class MNISTClassifier(pl.LightningModule):
        def __init__(self, lr=1e-3, feature_dim=128):
            torch.manual_seed(421)
            super(MNISTClassifier, self).__init__()
            self.save_hyperparameters()
    
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 10),
                nn.ReLU(),
            )
            self.lr = lr
            self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
            self.eval_loss = []
            self.eval_accuracy = []
            self.test_accuracy = []
            pl.seed_everything(888)
    
        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.linear_relu_stack(x)
            return x
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            self.log("train_loss", loss)
            return loss
    
        def validation_step(self, val_batch, batch_idx):
            loss, acc = self._shared_eval(val_batch)
            self.log("val_accuracy", acc)
            self.eval_loss.append(loss)
            self.eval_accuracy.append(acc)
            return {"val_loss": loss, "val_accuracy": acc}
    
        def test_step(self, test_batch, batch_idx):
            loss, acc = self._shared_eval(test_batch)
            self.test_accuracy.append(acc)
            self.log("test_accuracy", acc, sync_dist=True, on_epoch=True)
            return {"test_loss": loss, "test_accuracy": acc}
    
        def _shared_eval(self, batch):
            x, y = batch
            logits = self.forward(x)
            loss = F.nll_loss(logits, y)
            acc = self.accuracy(logits, y)
            return loss, acc
    
        def on_validation_epoch_end(self):
            avg_loss = torch.stack(self.eval_loss).mean()
            avg_acc = torch.stack(self.eval_accuracy).mean()
            self.log("val_loss", avg_loss, sync_dist=True)
            self.log("val_accuracy", avg_acc, sync_dist=True)
            self.eval_loss.clear()
            self.eval_accuracy.clear()
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer
    

You don’t need to modify the definition of the PyTorch Lightning model or datamodule.

## Define a training function

This code defines a training function for each worker. Comparing the training function with the original PyTorch Lightning code, notice three main differences:

  * Distributed strategy: Use `RayDDPStrategy`.

  * Cluster environment: Use `RayLightningEnvironment`.

  * Parallel devices: Always set to `devices="auto"` to use all available devices configured by `TorchTrainer`.

See Getting Started with PyTorch Lightning for more information.

For checkpoint reporting, Ray Train provides a minimal `RayTrainReportCallback` class that reports metrics and checkpoints at the end of each train epoch. For more complex checkpoint logic, implement custom callbacks. See Saving and Loading Checkpoint.
    
    
    use_gpu = True # Set to False if you want to run without GPUs
    num_workers = 4
    
    
    
    import pytorch_lightning as pl
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    
    def train_func_per_worker():
        model = MNISTClassifier(lr=1e-3, feature_dim=128)
        datamodule = MNISTDataModule(batch_size=128)
    
        trainer = pl.Trainer(
            devices="auto",
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            callbacks=[RayTrainReportCallback()],
            max_epochs=10,
            accelerator="gpu" if use_gpu else "cpu",
            log_every_n_steps=100,
            logger=CSVLogger("logs"),
        )
        
        trainer = prepare_trainer(trainer)
        
        # Train model
        trainer.fit(model, datamodule=datamodule)
    
        # Evaluation on the test dataset
        trainer.test(model, datamodule=datamodule)
    

Now put everything together:
    
    
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    
    run_config = RunConfig(
        name="ptl-mnist-example",
        storage_path="/tmp/ray_results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_accuracy",
            checkpoint_score_order="max",
        ),
    )
    
    trainer = TorchTrainer(
        train_func_per_worker,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    

Now fit your trainer:
    
    
    result = trainer.fit()
    

### Tune Status

Current time:| 2023-08-07 23:41:11  
---|---  
Running for: | 00:00:39.80   
Memory: | 24.2/186.6 GiB   
  
### System Info

Using FIFO scheduling algorithm.  
Logical resource usage: 1.0/48 CPUs, 4.0/4 GPUs 

### Trial Status

Trial name | status | loc |  iter|  total time (s)|  train_loss|  val_accuracy|  val_loss  
---|---|---|---|---|---|---|---  
TorchTrainer_78346_00000| TERMINATED| 10.0.6.244:120026|  10|  29.0221|  0.0315938|  0.970002|  -12.3466  
      
    
    (TorchTrainer pid=120026) Starting distributed worker processes: ['120176 (10.0.6.244)', '120177 (10.0.6.244)', '120178 (10.0.6.244)', '120179 (10.0.6.244)']
    (RayTrainWorker pid=120176) Setting up process group for: env:// [rank=0, world_size=4]
    (RayTrainWorker pid=120176) [rank: 0] Global seed set to 888
    (RayTrainWorker pid=120176) GPU available: True (cuda), used: True
    (RayTrainWorker pid=120176) TPU available: False, using: 0 TPU cores
    (RayTrainWorker pid=120176) IPU available: False, using: 0 IPUs
    (RayTrainWorker pid=120176) HPU available: False, using: 0 HPUs
    
    
    
    (RayTrainWorker pid=120178) Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    (RayTrainWorker pid=120178) Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_2/MNIST/raw/train-images-idx3-ubyte.gz
    
    
    
      0%|          | 0/9912422 [00:00<?, ?it/s]
    100%|██████████| 9912422/9912422 [00:00<00:00, 94562894.32it/s]
      9%|▉         | 917504/9912422 [00:00<00:00, 9166590.91it/s]
    100%|██████████| 9912422/9912422 [00:00<00:00, 115619443.32it/s]
    
    
    
    (RayTrainWorker pid=120179) Extracting /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_3/MNIST/raw/train-images-idx3-ubyte.gz to /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_3/MNIST/raw
    (RayTrainWorker pid=120176) 
    
    
    
    (RayTrainWorker pid=120177) Missing logger folder: logs/lightning_logs
    (RayTrainWorker pid=120176) LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
    (RayTrainWorker pid=120176)   | Name              | Type               | Params
    (RayTrainWorker pid=120176) ---------------------------------------------------------
    (RayTrainWorker pid=120176) 0 | linear_relu_stack | Sequential         | 101 K 
    (RayTrainWorker pid=120176) 1 | accuracy          | MulticlassAccuracy | 0     
    (RayTrainWorker pid=120176) ---------------------------------------------------------
    (RayTrainWorker pid=120176) 101 K     Trainable params
    (RayTrainWorker pid=120176) 0         Non-trainable params
    (RayTrainWorker pid=120176) 101 K     Total params
    (RayTrainWorker pid=120176) 0.407     Total estimated model params size (MB)
    
    
    
    Sanity Checking: 0it [00:00, ?it/s]) 
    Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]
    Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:00<00:00,  2.69it/s]
    
    
    
    (RayTrainWorker pid=120176) /mnt/cluster_storage/pypi/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:432: PossibleUserWarning: It is recommended to use `self.log('val_accuracy', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
    (RayTrainWorker pid=120176)   warning_cache.warn(
    (RayTrainWorker pid=120179) [rank: 3] Global seed set to 888 [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)
    
    
    
    Epoch 0:   0%|          | 0/108 [00:00<?, ?it/s] 
    Epoch 0:  12%|█▏        | 13/108 [00:00<00:02, 39.35it/s, v_num=0]
    Epoch 0:  25%|██▌       | 27/108 [00:00<00:01, 59.26it/s, v_num=0]
    Epoch 0:  26%|██▌       | 28/108 [00:00<00:01, 61.03it/s, v_num=0]
    Epoch 0:  27%|██▋       | 29/108 [00:00<00:01, 62.76it/s, v_num=0]
    Epoch 0:  42%|████▏     | 45/108 [00:00<00:00, 81.02it/s, v_num=0]
    Epoch 0:  53%|█████▎    | 57/108 [00:00<00:00, 86.01it/s, v_num=0]
    Epoch 0:  64%|██████▍   | 69/108 [00:00<00:00, 88.63it/s, v_num=0]
    Epoch 0:  81%|████████  | 87/108 [00:00<00:00, 98.04it/s, v_num=0]
    Epoch 0:  81%|████████▏ | 88/108 [00:00<00:00, 98.69it/s, v_num=0]
    Epoch 0:  82%|████████▏ | 89/108 [00:00<00:00, 99.34it/s, v_num=0]
    Epoch 0:  96%|█████████▋| 104/108 [00:00<00:00, 104.14it/s, v_num=0]
    Epoch 0:  97%|█████████▋| 105/108 [00:01<00:00, 104.71it/s, v_num=0]
    Epoch 0:  98%|█████████▊| 106/108 [00:01<00:00, 105.22it/s, v_num=0]
    Epoch 0: 100%|██████████| 108/108 [00:01<00:00, 105.79it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 171.69it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 200.99it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 221.66it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 215.50it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 194.14it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 205.63it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 215.27it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 216.26it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 198.67it/s]
    Validation DataLoader 0: 100%|██████████| 10/10 [00:00<00:00, 205.79it/s]
    Epoch 0: 100%|██████████| 108/108 [00:01<00:00, 79.84it/s, v_num=0] 
                                                                             
    Epoch 1:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 1:  11%|█         | 12/108 [00:00<00:02, 32.36it/s, v_num=0]
    Epoch 1:  23%|██▎       | 25/108 [00:00<00:01, 50.16it/s, v_num=0]
    Epoch 1:  37%|███▋      | 40/108 [00:00<00:01, 65.95it/s, v_num=0]
    Epoch 1:  38%|███▊      | 41/108 [00:00<00:00, 67.05it/s, v_num=0]
    Epoch 1:  50%|█████     | 54/108 [00:00<00:00, 75.52it/s, v_num=0]
    Epoch 1:  51%|█████     | 55/108 [00:00<00:00, 76.40it/s, v_num=0]
    Epoch 1:  62%|██████▏   | 67/108 [00:00<00:00, 81.72it/s, v_num=0]
    (RayTrainWorker pid=120178) Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz [repeated 15x across cluster]
    Epoch 1:  77%|███████▋  | 83/108 [00:00<00:00, 89.48it/s, v_num=0]
    (RayTrainWorker pid=120178) Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_2/MNIST/raw/t10k-labels-idx1-ubyte.gz [repeated 15x across cluster]
    Epoch 1:  78%|███████▊  | 84/108 [00:00<00:00, 89.21it/s, v_num=0]
    Epoch 1:  91%|█████████ | 98/108 [00:01<00:00, 93.27it/s, v_num=0]
    (RayTrainWorker pid=120178) Extracting /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_2/MNIST/raw/t10k-labels-idx1-ubyte.gz to /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/rank_2/MNIST/raw [repeated 15x across cluster]
    Epoch 1:  92%|█████████▏| 99/108 [00:01<00:00, 93.94it/s, v_num=0]
    Epoch 1:  93%|█████████▎| 100/108 [00:01<00:00, 94.57it/s, v_num=0]
    Epoch 1: 100%|██████████| 108/108 [00:01<00:00, 98.06it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 320.27it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 291.99it/s]
    (RayTrainWorker pid=120176)  [repeated 19x across cluster]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 291.61it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 268.90it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 290.07it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 293.52it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 299.70it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 304.80it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 310.16it/s]
    Validation DataLoader 0: 100%|██████████| 10/10 [00:00<00:00, 303.63it/s]
    Epoch 1: 100%|██████████| 108/108 [00:01<00:00, 76.12it/s, v_num=0]
                                                                             
    Epoch 2:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 2:   7%|▋         | 8/108 [00:00<00:03, 25.23it/s, v_num=0]
    Epoch 2:  16%|█▌        | 17/108 [00:00<00:02, 39.73it/s, v_num=0]
    Epoch 2:  17%|█▋        | 18/108 [00:00<00:02, 41.60it/s, v_num=0]
    Epoch 2:  18%|█▊        | 19/108 [00:00<00:02, 43.49it/s, v_num=0]
    Epoch 2:  18%|█▊        | 19/108 [00:00<00:02, 43.46it/s, v_num=0]
    Epoch 2:  19%|█▊        | 20/108 [00:00<00:01, 45.27it/s, v_num=0]
    Epoch 2:  27%|██▋       | 29/108 [00:00<00:01, 53.08it/s, v_num=0]
    Epoch 2:  42%|████▏     | 45/108 [00:00<00:00, 69.12it/s, v_num=0]
    Epoch 2:  43%|████▎     | 46/108 [00:00<00:00, 70.31it/s, v_num=0]
    Epoch 2:  44%|████▎     | 47/108 [00:00<00:00, 71.51it/s, v_num=0]
    Epoch 2:  44%|████▍     | 48/108 [00:00<00:00, 72.71it/s, v_num=0]
    Epoch 2:  59%|█████▉    | 64/108 [00:00<00:00, 83.97it/s, v_num=0]
    Epoch 2:  75%|███████▌  | 81/108 [00:00<00:00, 93.77it/s, v_num=0]
    Epoch 2:  90%|████████▉ | 97/108 [00:00<00:00, 99.35it/s, v_num=0]
    Epoch 2: 100%|██████████| 108/108 [00:01<00:00, 101.71it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 212.13it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 184.45it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 228.42it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 225.00it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 250.65it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 251.36it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 268.85it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 256.15it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 269.87it/s]
    Epoch 2: 100%|██████████| 108/108 [00:01<00:00, 77.52it/s, v_num=0] it/s]
                                                                             
    Epoch 3:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 3:   8%|▊         | 9/108 [00:00<00:03, 25.68it/s, v_num=0]
    Epoch 3:   9%|▉         | 10/108 [00:00<00:03, 28.26it/s, v_num=0]
    Epoch 3:  20%|██        | 22/108 [00:00<00:01, 48.10it/s, v_num=0]
    Epoch 3:  21%|██▏       | 23/108 [00:00<00:01, 49.73it/s, v_num=0]
    Epoch 3:  22%|██▏       | 24/108 [00:00<00:01, 51.34it/s, v_num=0]
    Epoch 3:  23%|██▎       | 25/108 [00:00<00:01, 52.98it/s, v_num=0]
    Epoch 3:  37%|███▋      | 40/108 [00:00<00:00, 69.67it/s, v_num=0]
    Epoch 3:  51%|█████     | 55/108 [00:00<00:00, 80.93it/s, v_num=0]
    Epoch 3:  64%|██████▍   | 69/108 [00:00<00:00, 87.15it/s, v_num=0]
    Epoch 3:  65%|██████▍   | 70/108 [00:00<00:00, 88.04it/s, v_num=0]
    Epoch 3:  66%|██████▌   | 71/108 [00:00<00:00, 88.92it/s, v_num=0]
    Epoch 3:  77%|███████▋  | 83/108 [00:00<00:00, 92.62it/s, v_num=0]
    Epoch 3:  86%|████████▌ | 93/108 [00:01<00:00, 92.33it/s, v_num=0]
    Epoch 3:  87%|████████▋ | 94/108 [00:01<00:00, 92.93it/s, v_num=0]
    Epoch 3:  88%|████████▊ | 95/108 [00:01<00:00, 93.61it/s, v_num=0]
    Epoch 3: 100%|██████████| 108/108 [00:01<00:00, 97.43it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 308.50it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 344.87it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 375.98it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 335.26it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 327.34it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 317.66it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 332.79it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 188.14it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 201.21it/s]
    Epoch 3: 100%|██████████| 108/108 [00:01<00:00, 75.94it/s, v_num=0]6it/s]
                                                                             
    Epoch 4:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 4:  10%|█         | 11/108 [00:00<00:03, 30.09it/s, v_num=0]
    Epoch 4:  20%|██        | 22/108 [00:00<00:01, 46.96it/s, v_num=0]
    Epoch 4:  21%|██▏       | 23/108 [00:00<00:01, 47.88it/s, v_num=0]
    Epoch 4:  35%|███▌      | 38/108 [00:00<00:01, 65.26it/s, v_num=0]
    Epoch 4:  36%|███▌      | 39/108 [00:00<00:01, 65.73it/s, v_num=0]
    Epoch 4:  53%|█████▎    | 57/108 [00:00<00:00, 81.51it/s, v_num=0]
    Epoch 4:  54%|█████▎    | 58/108 [00:00<00:00, 82.56it/s, v_num=0]
    Epoch 4:  68%|██████▊   | 73/108 [00:00<00:00, 89.69it/s, v_num=0]
    Epoch 4:  69%|██████▊   | 74/108 [00:00<00:00, 90.53it/s, v_num=0]
    Epoch 4:  83%|████████▎ | 90/108 [00:00<00:00, 98.32it/s, v_num=0]
    Epoch 4:  98%|█████████▊| 106/108 [00:01<00:00, 103.12it/s, v_num=0]
    Epoch 4: 100%|██████████| 108/108 [00:01<00:00, 103.78it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 268.49it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 298.62it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 282.88it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 256.50it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 276.28it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 268.05it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 276.18it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 290.08it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 261.92it/s]
    Validation DataLoader 0: 100%|██████████| 10/10 [00:00<00:00, 274.00it/s]
    Epoch 4: 100%|██████████| 108/108 [00:01<00:00, 78.54it/s, v_num=0] 
                                                                             
    Epoch 5:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 5:   5%|▍         | 5/108 [00:00<00:06, 15.52it/s, v_num=0]
    (RayTrainWorker pid=120176)  [repeated 9x across cluster]
    Epoch 5:  17%|█▋        | 18/108 [00:00<00:02, 42.53it/s, v_num=0]
    Epoch 5:  26%|██▌       | 28/108 [00:00<00:01, 52.36it/s, v_num=0]
    Epoch 5:  27%|██▋       | 29/108 [00:00<00:01, 53.91it/s, v_num=0]
    Epoch 5:  28%|██▊       | 30/108 [00:00<00:01, 55.45it/s, v_num=0]
    Epoch 5:  29%|██▊       | 31/108 [00:00<00:01, 56.96it/s, v_num=0]
    Epoch 5:  37%|███▋      | 40/108 [00:00<00:01, 61.48it/s, v_num=0]
    Epoch 5:  38%|███▊      | 41/108 [00:00<00:01, 62.61it/s, v_num=0]
    Epoch 5:  39%|███▉      | 42/108 [00:00<00:01, 63.79it/s, v_num=0]
    Epoch 5:  40%|███▉      | 43/108 [00:00<00:01, 64.96it/s, v_num=0]
    Epoch 5:  48%|████▊     | 52/108 [00:00<00:00, 67.96it/s, v_num=0]
    Epoch 5:  49%|████▉     | 53/108 [00:00<00:00, 68.88it/s, v_num=0]
    Epoch 5:  50%|█████     | 54/108 [00:00<00:00, 69.77it/s, v_num=0]
    Epoch 5:  62%|██████▏   | 67/108 [00:00<00:00, 76.43it/s, v_num=0]
    Epoch 5:  78%|███████▊  | 84/108 [00:00<00:00, 85.56it/s, v_num=0]
    Epoch 5:  79%|███████▊  | 85/108 [00:00<00:00, 86.17it/s, v_num=0]
    Epoch 5:  93%|█████████▎| 100/108 [00:01<00:00, 92.27it/s, v_num=0]
    Epoch 5: 100%|██████████| 108/108 [00:01<00:00, 94.81it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 255.91it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 206.50it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 214.91it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 236.35it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 240.05it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 234.90it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 240.78it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 252.00it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 255.18it/s]
    Epoch 5: 100%|██████████| 108/108 [00:01<00:00, 72.36it/s, v_num=0]1it/s]
                                                                             
    Epoch 6:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 6:  15%|█▍        | 16/108 [00:00<00:02, 44.27it/s, v_num=0]
    Epoch 6:  25%|██▌       | 27/108 [00:00<00:01, 57.21it/s, v_num=0]
    Epoch 6:  38%|███▊      | 41/108 [00:00<00:00, 70.86it/s, v_num=0]
    Epoch 6:  39%|███▉      | 42/108 [00:00<00:00, 71.82it/s, v_num=0]
    Epoch 6:  55%|█████▍    | 59/108 [00:00<00:00, 85.97it/s, v_num=0]
    Epoch 6:  68%|██████▊   | 73/108 [00:00<00:00, 91.53it/s, v_num=0]
    Epoch 6:  81%|████████  | 87/108 [00:00<00:00, 96.88it/s, v_num=0]
    Epoch 6:  92%|█████████▏| 99/108 [00:00<00:00, 99.33it/s, v_num=0]
    Epoch 6:  93%|█████████▎| 100/108 [00:01<00:00, 98.66it/s, v_num=0]
    Epoch 6:  94%|█████████▎| 101/108 [00:01<00:00, 99.34it/s, v_num=0]
    Epoch 6: 100%|██████████| 108/108 [00:01<00:00, 102.79it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 197.51it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 143.68it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 156.17it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 180.52it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 205.25it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 212.20it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 195.64it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 211.21it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 225.13it/s]
    Epoch 6: 100%|██████████| 108/108 [00:01<00:00, 76.04it/s, v_num=0] it/s]
                                                                             
    Epoch 7:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 7:  11%|█         | 12/108 [00:00<00:02, 33.31it/s, v_num=0]
    Epoch 7:  20%|██        | 22/108 [00:00<00:01, 46.90it/s, v_num=0]
    Epoch 7:  22%|██▏       | 24/108 [00:00<00:01, 50.49it/s, v_num=0]
    Epoch 7:  31%|███▏      | 34/108 [00:00<00:01, 58.20it/s, v_num=0]
    Epoch 7:  32%|███▏      | 35/108 [00:00<00:01, 59.59it/s, v_num=0]
    Epoch 7:  33%|███▎      | 36/108 [00:00<00:01, 60.97it/s, v_num=0]
    Epoch 7:  48%|████▊     | 52/108 [00:00<00:00, 74.69it/s, v_num=0]
    Epoch 7:  64%|██████▍   | 69/108 [00:00<00:00, 85.96it/s, v_num=0]
    Epoch 7:  80%|███████▉  | 86/108 [00:00<00:00, 94.41it/s, v_num=0]
    Epoch 7:  81%|████████▏ | 88/108 [00:00<00:00, 95.91it/s, v_num=0]
    Epoch 7:  97%|█████████▋| 105/108 [00:01<00:00, 102.61it/s, v_num=0]
    Epoch 7: 100%|██████████| 108/108 [00:01<00:00, 103.00it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 215.46it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 246.46it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 264.39it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 256.84it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 218.46it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 230.90it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 243.53it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 253.83it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 249.22it/s]
    Epoch 7: 100%|██████████| 108/108 [00:01<00:00, 78.36it/s, v_num=0] it/s]
                                                                             
    Epoch 8:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 8:   7%|▋         | 8/108 [00:00<00:03, 25.72it/s, v_num=0]
    Epoch 8:  19%|█▊        | 20/108 [00:00<00:01, 47.54it/s, v_num=0]
    Epoch 8:  31%|███       | 33/108 [00:00<00:01, 62.61it/s, v_num=0]
    (RayTrainWorker pid=120176)  [repeated 7x across cluster]
    Epoch 8:  44%|████▎     | 47/108 [00:00<00:00, 74.45it/s, v_num=0]
    Epoch 8:  55%|█████▍    | 59/108 [00:00<00:00, 81.84it/s, v_num=0]
    Epoch 8:  56%|█████▌    | 60/108 [00:00<00:00, 80.82it/s, v_num=0]
    Epoch 8:  57%|█████▋    | 62/108 [00:00<00:00, 82.74it/s, v_num=0]
    Epoch 8:  58%|█████▊    | 63/108 [00:00<00:00, 83.69it/s, v_num=0]
    Epoch 8:  70%|███████   | 76/108 [00:00<00:00, 88.60it/s, v_num=0]
    Epoch 8:  85%|████████▌ | 92/108 [00:00<00:00, 96.53it/s, v_num=0]
    Epoch 8:  86%|████████▌ | 93/108 [00:00<00:00, 96.21it/s, v_num=0]
    Epoch 8:  87%|████████▋ | 94/108 [00:00<00:00, 96.72it/s, v_num=0]
    Epoch 8:  88%|████████▊ | 95/108 [00:00<00:00, 97.32it/s, v_num=0]
    Epoch 8:  89%|████████▉ | 96/108 [00:00<00:00, 98.03it/s, v_num=0]
    Epoch 8: 100%|██████████| 108/108 [00:01<00:00, 102.15it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 228.96it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 220.63it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 220.41it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 208.74it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 221.74it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 243.64it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 253.60it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 254.93it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 207.23it/s]
    Epoch 8: 100%|██████████| 108/108 [00:01<00:00, 78.28it/s, v_num=0] it/s]
                                                                             
    Epoch 9:   0%|          | 0/108 [00:00<?, ?it/s, v_num=0]          
    Epoch 9:  11%|█         | 12/108 [00:00<00:02, 33.03it/s, v_num=0]
    Epoch 9:  21%|██▏       | 23/108 [00:00<00:01, 48.82it/s, v_num=0]
    Epoch 9:  31%|███       | 33/108 [00:00<00:01, 58.62it/s, v_num=0]
    Epoch 9:  31%|███▏      | 34/108 [00:00<00:01, 58.61it/s, v_num=0]
    Epoch 9:  32%|███▏      | 35/108 [00:00<00:01, 59.89it/s, v_num=0]
    Epoch 9:  33%|███▎      | 36/108 [00:00<00:01, 61.11it/s, v_num=0]
    Epoch 9:  46%|████▋     | 50/108 [00:00<00:00, 71.95it/s, v_num=0]
    Epoch 9:  61%|██████    | 66/108 [00:00<00:00, 82.62it/s, v_num=0]
    Epoch 9:  70%|███████   | 76/108 [00:00<00:00, 83.77it/s, v_num=0]
    Epoch 9:  71%|███████▏  | 77/108 [00:00<00:00, 84.54it/s, v_num=0]
    Epoch 9:  72%|███████▏  | 78/108 [00:00<00:00, 85.33it/s, v_num=0]
    Epoch 9:  88%|████████▊ | 95/108 [00:01<00:00, 93.18it/s, v_num=0]
    Epoch 9: 100%|██████████| 108/108 [00:01<00:00, 98.27it/s, v_num=0]
    Validation: 0it [00:00, ?it/s]76) 
    Validation:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]
    Validation DataLoader 0:  10%|█         | 1/10 [00:00<00:00, 305.42it/s]
    Validation DataLoader 0:  20%|██        | 2/10 [00:00<00:00, 337.39it/s]
    Validation DataLoader 0:  30%|███       | 3/10 [00:00<00:00, 368.65it/s]
    Validation DataLoader 0:  40%|████      | 4/10 [00:00<00:00, 361.22it/s]
    Validation DataLoader 0:  50%|█████     | 5/10 [00:00<00:00, 250.96it/s]
    Validation DataLoader 0:  60%|██████    | 6/10 [00:00<00:00, 271.98it/s]
    Validation DataLoader 0:  70%|███████   | 7/10 [00:00<00:00, 289.64it/s]
    Validation DataLoader 0:  80%|████████  | 8/10 [00:00<00:00, 304.16it/s]
    Validation DataLoader 0:  90%|█████████ | 9/10 [00:00<00:00, 184.87it/s]
    Validation DataLoader 0: 100%|██████████| 10/10 [00:00<00:00, 196.99it/s]
    Epoch 9: 100%|██████████| 108/108 [00:01<00:00, 74.63it/s, v_num=0]
                                                                             
    
    
    
    (RayTrainWorker pid=120176) `Trainer.fit` stopped: `max_epochs=10` reached.
    100%|██████████| 4542/4542 [00:00<00:00, 48474627.91it/s] [repeated 14x across cluster]
    100%|██████████| 9912422/9912422 [00:00<00:00, 90032420.31it/s] [repeated 2x across cluster]
    (RayTrainWorker pid=120176)  [repeated 5x across cluster]
    (RayTrainWorker pid=120178) Missing logger folder: logs/lightning_logs [repeated 2x across cluster]
    (RayTrainWorker pid=120179) LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3] [repeated 3x across cluster]
    (RayTrainWorker pid=120176) [rank: 0] Global seed set to 888
    
    
    
    Epoch 9: 100%|██████████| 108/108 [00:01<00:00, 66.61it/s, v_num=0]
    
    
    
    (RayTrainWorker pid=120176) /mnt/cluster_storage/pypi/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:225: PossibleUserWarning: Using `DistributedSampler` with the dataloaders. During `trainer.test()`, it is recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.
    (RayTrainWorker pid=120176)   rank_zero_warn(
    
    
    
    Testing DataLoader 0:  25%|██▌       | 5/20 [00:00<00:00, 146.57it/s]
    Testing DataLoader 0: 100%|██████████| 20/20 [00:00<00:00, 163.98it/s]
    Testing DataLoader 0: 100%|██████████| 20/20 [00:00<00:00, 125.34it/s]
    (RayTrainWorker pid=120176) ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    (RayTrainWorker pid=120176) ┃        Test metric        ┃       DataLoader 0        ┃
    (RayTrainWorker pid=120176) ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    (RayTrainWorker pid=120176) │       test_accuracy       │    0.9740999937057495     │
    (RayTrainWorker pid=120176) └───────────────────────────┴───────────────────────────┘
    
    
    
    2023-08-07 23:41:11,072	INFO tune.py:1145 -- Total run time: 39.92 seconds (39.80 seconds for the tuning loop).
    

## Check training results and checkpoints
    
    
    result
    
    
    
    Result(
      metrics={'train_loss': 0.03159375861287117, 'val_accuracy': 0.9700015783309937, 'val_loss': -12.346583366394043, 'epoch': 9, 'step': 1080},
      path='/tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31',
      checkpoint=LegacyTorchCheckpoint(local_path=/tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31/checkpoint_000009)
    )
    
    
    
    print("Validation Accuracy: ", result.metrics["val_accuracy"])
    print("Trial Directory: ", result.path)
    print(sorted(os.listdir(result.path)))
    
    
    
    Validation Accuracy:  0.9700015783309937
    Trial Directory:  /tmp/ray_results/ptl-mnist-example/TorchTrainer_78346_00000_0_2023-08-07_23-40-31
    ['checkpoint_000007', 'checkpoint_000008', 'checkpoint_000009', 'events.out.tfevents.1691476838.ip-10-0-6-244', 'params.json', 'params.pkl', 'progress.csv', 'rank_0', 'rank_0.lock', 'rank_1', 'rank_1.lock', 'rank_2', 'rank_2.lock', 'rank_3', 'rank_3.lock', 'result.json']
    

Ray Train saved three checkpoints(`checkpoint_000007`, `checkpoint_000008`, `checkpoint_000009`) in the trial directory. The following code retrieves the latest checkpoint from the fit results and loads it back into the model.

If you lost the in-memory result object, you can restore the model from the checkpoint file. The checkpoint path is: `/tmp/ray_results/ptl-mnist-example/TorchTrainer_eb925_00000_0_2023-08-07_23-15-06/checkpoint_000009/checkpoint.ckpt`.
    
    
    checkpoint = result.checkpoint
    
    with checkpoint.as_directory() as ckpt_dir:
        best_model = MNISTClassifier.load_from_checkpoint(f"{ckpt_dir}/checkpoint.ckpt")
    
    best_model
    
    
    
    Global seed set to 888
    
    
    
    MNISTClassifier(
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=10, bias=True)
        (3): ReLU()
      )
      (accuracy): MulticlassAccuracy()
    )
    

## See also

  * Getting Started with PyTorch Lightning for a tutorial on using Ray Train and PyTorch Lightning

  * [Ray Train Examples](../../examples.md) for more use cases