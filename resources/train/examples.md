# Examples

---

# Train Examples

Below are examples for using Ray Train with a variety of frameworks and use cases. Ray Train makes it easy to scale out each of these examples to a large cluster of GPUs.

## Beginner

Framework| Example  
---|---  
PyTorch| [Distributing your PyTorch Training Code with Ray Train and Ray Data](examples/pytorch/distributing-pytorch/README.md)  
Lightning| [Train an image classifier with Lightning](examples/lightning/lightning_mnist_example.md)  
Accelerate, PyTorch, Hugging Face| [Train a text classifier with Hugging Face Accelerate](examples/accelerate/accelerate_example.md)  
TensorFlow| [Train an image classifier with TensorFlow](examples/tf/tensorflow_mnist_example.md)  
Horovod| [Train with Horovod and PyTorch](examples/horovod/horovod_example.md)  
PyTorch| [Train ResNet model with Intel Gaudi](examples/intel_gaudi/resnet.md)  
Transformers| [Train BERT model with Intel Gaudi](examples/intel_gaudi/bert.md)  
PyTorch| Profiling a Ray Train Workload with PyTorch Profiler  
XGBoost| Train a tabular model with XGBoost  
  
## Intermediate

Framework| Example  
---|---  
PyTorch| [Get started with PyTorch Fully Sharded Data Parallel (FSDP2) and Ray Train](examples/pytorch/pytorch-fsdp/README.md)  
PyTorch, DeepSpeed| [Fine-tune an LLM with Ray Train and DeepSpeed](examples/pytorch/deepspeed_finetune/README.md)  
DeepSpeed, PyTorch| [Train a text classifier with DeepSpeed](examples/deepspeed/deepspeed_example.md)  
PyTorch| [Fine-tune a personalized Stable Diffusion model](examples/pytorch/dreambooth_finetuning.md)  
Accelerate, Transformers| [Finetune Stable Diffusion and generate images with Intel Gaudi](examples/intel_gaudi/sd.md)  
Lightning| [Train a text classifier with PyTorch Lightning and Ray Data](examples/lightning/lightning_cola_advanced.md)  
Transformers| [Train a text classifier with Hugging Face Transformers](examples/transformers/huggingface_text_classification.md)  
Accelerate, Transformers| [Fine-tune Llama-2-7b and Llama-2-70b with Intel Gaudi](examples/intel_gaudi/llama.md)  
Accelerate, Transformers, DeepSpeed| [Pre-train Llama-2 with Intel Gaudi](examples/intel_gaudi/llama_pretrain.md)  
  
## Advanced

Framework| Example  
---|---  
PyTorch, AWS Neuron| [Fine-tune Llama3.1 with AWS Trainium](examples/aws-trainium/llama3.md)  
Accelerate, DeepSpeed, Hugging Face| [Fine-tune a Llama-2 text generation model with DeepSpeed and Hugging Face Accelerate](https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed)  
Hugging Face, DeepSpeed| [Fine-tune a GPT-J-6B text generation model with DeepSpeed and Hugging Face Transformers](examples/deepspeed/gptj_deepspeed_fine_tuning.md)  
Lightning, DeepSpeed| [Fine-tune a vicuna-13b text generation model with PyTorch Lightning and DeepSpeed](examples/lightning/vicuna_13b_lightning_deepspeed_finetune.md)  
Lightning| Fine-tune a dolly-v2-7b text generation model with PyTorch Lightning and FSDP