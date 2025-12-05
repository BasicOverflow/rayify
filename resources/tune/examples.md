# Ray Tune Examples

---

# Ray Tune Examples

Tip

See [Ray Tune: Hyperparameter Tuning](../index.md#tune-main) to learn more about Tune features.

Below are examples for using Ray Tune for a variety of use cases and sorted by categories:

  * ML frameworks

  * Experiment tracking tools

  * Hyperparameter optimization frameworks

  * Others

  * Exercises

## ML frameworks

Ray Tune integrates with many popular machine learning frameworks. Here you find a few practical examples showing you how to tune your models. At the end of these guides you will often find links to even more examples.

How to use Tune with Keras and TensorFlow models  
---  
How to use Tune with PyTorch models  
How to tune PyTorch Lightning models  
Tuning RL experiments with Ray Tune and Ray Serve  
Tuning XGBoost parameters with Tune  
Tuning LightGBM parameters with Tune  
Tuning Hugging Face Transformers with Tune  
  
## Experiment tracking tools

Ray Tune integrates with some popular Experiment tracking and management tools, such as CometML, or Weights & Biases. For how to use Ray Tune with Tensorboard, see Guide to logging and outputs.

Using Aim with Ray Tune for experiment management  
---  
Using Comet with Ray Tune for experiment management  
Tracking your experiment process Weights & Biases  
Using MLflow tracking and auto logging with Tune  
  
## Hyperparameter optimization frameworks

Tune integrates with a wide variety of hyperparameter optimization frameworks and their respective search algorithms. See the following detailed examples for each integration:

Running Tune experiments with AxSearch  
---  
Running Tune experiments with HyperOpt  
Running Tune experiments with BayesOpt  
Running Tune experiments with BOHB  
Running Tune experiments with Nevergrad  
Running Tune experiments with Optuna  
  
## Others

Simple example for doing a basic random and grid search  
---  
Example of using a simple tuning function with AsyncHyperBandScheduler  
Example of using a trainable function with HyperBandScheduler and the AsyncHyperBandScheduler  
PBT Function Example  
PB2 Example  
Logging Example  
  
## Exercises

Learn how to use Tune in your browser with the following Colab-based exercises.

Description | Library | Colab link  
---|---|---  
Basics of using Tune | PyTorch |   
Using search algorithms and trial schedulers to optimize your model | PyTorch |   
Using Population-Based Training (PBT) | PyTorch |   
Fine-tuning Hugging Face Transformers with PBT | Hugging Face Transformers and PyTorch |   
Logging Tune runs to Comet ML | Comet |   
  
Tutorial source files are on [GitHub](https://github.com/ray-project/tutorial).