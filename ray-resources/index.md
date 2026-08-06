# Welcome to Ray!

---

# Welcome to Ray

An open source framework to build and scale your ML and Python applications easily 

    [ ](ray-overview/examples.md)

### Scale with Ray
    
    
      from typing import Dict
      import numpy as np
    
      import ray
    
      # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
      ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))
    
      # Step 2: Define a Predictor class for inference.
      class HuggingFacePredictor:
          def __init__(self):
              from transformers import pipeline
              # Initialize a pre-trained GPT2 Huggingface pipeline.
              self.model = pipeline("text-generation", model="gpt2")
    
          # Logic for inference on 1 batch of data.
          def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
              # Get the predictions from the input batch.
              predictions = self.model(
                  list(batch["data"]), max_length=20, num_return_sequences=1)
              # `predictions` is a list of length-one lists. For example:
              # [[{"generated_text": "output_1"}], ..., [{"generated_text": "output_2"}]]
              # Modify the output to get it into the following format instead:
              # ["output_1", "output_2"]
              batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
              return batch
    
      # Use 2 parallel actors for inference. Each actor predicts on a
      # different partition of data.
      # Step 3: Map the Predictor over the Dataset to get predictions.
      predictions = ds.map_batches(HuggingFacePredictor, compute=ray.data.ActorPoolStrategy(size=2))
      # Step 4: Show one prediction output.
      predictions.show(limit=1)
                
    

Learn more about Ray Data [Examples](data/examples.md)
    
    
      from ray.train import ScalingConfig
      from ray.train.torch import TorchTrainer
    
      # Step 1: Set up PyTorch model training as you normally would.
      def train_func():
          model = ...
          train_dataset = ...
          for epoch in range(num_epochs):
              ...  # model training logic
    
      # Step 2: Set up Ray's PyTorch Trainer to run on 32 GPUs.
      trainer = TorchTrainer(
          train_loop_per_worker=train_func,
          scaling_config=ScalingConfig(num_workers=32, use_gpu=True),
          datasets={"train": train_dataset},
      )
    
      # Step 3: Run distributed model training on 32 GPUs.
      result = trainer.fit()
              
    

Learn more about Ray Train [Examples](train/examples.md)
    
    
      from ray import tune
      from ray.train import ScalingConfig
      from ray.train.lightgbm import LightGBMTrainer
    
      train_dataset, eval_dataset = ...
    
      # Step 1: Set up Ray's LightGBM Trainer to train on 64 CPUs.
      trainer = LightGBMTrainer(
          ...
          scaling_config=ScalingConfig(num_workers=64),
          datasets={"train": train_dataset, "eval": eval_dataset},
      )
    
      # Step 2: Set up Ray Tuner to run 1000 trials.
      tuner = tune.Tuner(
          trainer=trainer,
          param_space=hyper_param_space,
          tune_config=tune.TuneConfig(num_samples=1000),
      )
    
      # Step 3: Run distributed HPO with 1000 trials; each trial runs on 64 CPUs.
      result_grid = tuner.fit()
              
    

Learn more about Ray Tune Examples
    
    
      from io import BytesIO
      from fastapi import FastAPI
      from fastapi.responses import Response
      import torch
    
      from ray import serve
      from ray.serve.handle import DeploymentHandle
    
    
      app = FastAPI()
    
    
      @serve.deployment(num_replicas=1)
      @serve.ingress(app)
      class APIIngress:
          def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
              self.handle = diffusion_model_handle
    
          @app.get(
              "/imagine",
              responses={200: {"content": {"image/png": {}}}},
              response_class=Response,
          )
          async def generate(self, prompt: str, img_size: int = 512):
              assert len(prompt), "prompt parameter cannot be empty"
    
              image = await self.handle.generate.remote(prompt, img_size=img_size)
              file_stream = BytesIO()
              image.save(file_stream, "PNG")
              return Response(content=file_stream.getvalue(), media_type="image/png")
    
    
      @serve.deployment(
          ray_actor_options={"num_gpus": 1},
          autoscaling_config={"min_replicas": 0, "max_replicas": 2},
      )
      class StableDiffusionV2:
          def __init__(self):
              from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
    
              model_id = "stabilityai/stable-diffusion-2"
    
              scheduler = EulerDiscreteScheduler.from_pretrained(
                  model_id, subfolder="scheduler"
              )
              self.pipe = StableDiffusionPipeline.from_pretrained(
                  model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
              )
              self.pipe = self.pipe.to("cuda")
    
          def generate(self, prompt: str, img_size: int = 512):
              assert len(prompt), "prompt parameter cannot be empty"
    
              with torch.autocast("cuda"):
                  image = self.pipe(prompt, height=img_size, width=img_size).images[0]
                  return image
    
    
      entrypoint = APIIngress.bind(StableDiffusionV2.bind())
              
    

Learn more about Ray Serve [Quickstart](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=scale_with_ray&redirectTo=/v2/template-preview/serve-stable-diffusion-v2)
    
    
      from ray.rllib.algorithms.ppo import PPOConfig
    
      # Step 1: Configure PPO to run 64 parallel workers to collect samples from the env.
      ppo_config = (
          PPOConfig()
          .environment(env="Taxi-v3")
          .rollouts(num_rollout_workers=64)
          .framework("torch")
          .training(model=rnn_lage)
      )
    
      # Step 2: Build the PPO algorithm.
      ppo_algo = ppo_config.build()
    
      # Step 3: Train and evaluate PPO.
      for _ in range(5):
          print(ppo_algo.train())
    
      ppo_algo.evaluate()
              
    

Learn more about Ray RLlib Examples

### Beyond the basics

#### Ray Libraries

Scale the entire ML pipeline from data ingest to model serving with high-level Python APIs that integrate with popular ecosystem frameworks. 

 Learn more 

#### Ray Core

Scale generic Python code with simple, foundational primitives that enable a high degree of control for building distributed applications or custom platforms. 

 Learn more 

#### Ray Clusters

Deploy a Ray cluster on AWS, GCP, Azure, or Kubernetes to seamlessly scale workloads for production. 

 Learn more 

