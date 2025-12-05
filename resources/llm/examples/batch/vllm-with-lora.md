# Batch Inference with LoRA Adapters

---

# Batch Inference with LoRA Adapters

In this example, we show how to perform batch inference using Ray Data LLM with LLM and a LoRA adapter.

To run this example, we need to install the following dependencies:
    
    
    pip install -qU "ray[llm]"
    
    
    
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    
    # 1. Construct a vLLM processor config.
    processor_config = vLLMEngineProcessorConfig(
        # The base model.
        model_source="unsloth/Llama-3.2-1B-Instruct",
        # vLLM engine config.
        engine_kwargs=dict(
            # Enable LoRA in the vLLM engine; otherwise you won't be able to
            # process requests with LoRA adapters.
            enable_lora=True,
            # You need to set the LoRA rank for the adapter.
            # The LoRA rank is the value of "r" in the LoRA config.
            # If you want to use multiple LoRA adapters in this pipeline,
            # please specify the maximum LoRA rank among all of them.
            max_lora_rank=32,
            # The maximum number of LoRA adapters vLLM cached. "1" means
            # vLLM only caches one LoRA adapter at a time, so if your dataset
            # needs more than one LoRA adapters, then there would be context
            # switching. On the other hand, while increasing max_loras reduces
            # the context switching, it increases the memory footprint.
            max_loras=1,
            # Older GPUs (e.g. T4) don't support bfloat16. You should remove
            # this line if you're using later GPUs.
            dtype="half",
            # Reduce the model length to fit small GPUs. You should remove
            # this line if you're using large GPUs.
            max_model_len=1024,
        ),
        # The batch size used in Ray Data.
        batch_size=16,
        # Use one GPU in this example.
        concurrency=1,
        # If you save the LoRA adapter in S3, you can set the following path.
        # dynamic_lora_loading_path="s3://your-lora-bucket/",
    )
    
    # 2. Construct a processor using the processor config.
    processor = build_llm_processor(
        processor_config,
        # Convert the input data to the OpenAI chat form.
        preprocess=lambda row: dict(
            # If you specify "model" in a request, and the model is different
            # from the model you specify in the processor config, then this
            # is the LoRA adapter. The "model" here can be a LoRA adapter
            # available in the HuggingFace Hub or a local path.
            #
            # If you set dynamic_lora_loading_path, then only specify the LoRA
            # path under dynamic_lora_loading_path.
            model="EdBergJr/Llama32_Baha_3",
            messages=[
                {"role": "system",
                 "content": "You are a calculator. Please only output the answer "
                    "of the given equation."},
                {"role": "user", "content": f"{row['id']} ** 3 = ?"},
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=20,
                detokenize=False,
            ),
        ),
        # Only keep the generated text in the output dataset.
        postprocess=lambda row: {
            "resp": row["generated_text"],
        },
    )
    
    # 3. Synthesize a dataset with 30 rows.
    ds = ray.data.range(30)
    # 4. Apply the processor to the dataset. Note that this line won't kick off
    # anything because processor is execution lazily.
    ds = processor(ds)
    # Materialization kicks off the pipeline execution.
    ds = ds.materialize()
    
    # 5. Print all outputs.
    for out in ds.take_all():
        print(out)
        print("==========")
    
    # 6. Shutdown Ray to release resources.
    ray.shutdown()