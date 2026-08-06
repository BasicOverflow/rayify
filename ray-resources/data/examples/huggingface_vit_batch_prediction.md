# Image Classification Batch Inference with Huggingface Vision Transformer

---

# Image Classification Batch Inference with Huggingface Vision Transformer

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=huggingface_vit_batch_prediction)   

In this example, we will introduce how to use [Ray Data](https://docs.ray.io/en/latest/data/data.html) for **large-scale image classification batch inference with multiple GPU workers.**

In particular, we will:

  * Load the [Imagenette](https://github.com/fastai/imagenette) dataset from an S3 bucket and create a `Ray Dataset`.

  * Load a pretrained Vision Transformer from Huggingface that’s been trained on [ImageNet](https://www.image-net.org/).

  * Use [Ray Data](https://docs.ray.io/en/latest/data/data.html) to preprocess the dataset and do model inference in parallel across multiple GPUs.

  * Evaluate the predictions and save results to S3/local disk.

_Note: This example will still work even if you do not have GPUs available, but overall performance will be slower._

To run this example, you will need to install the following:
    
    
    !pip install -q -U "ray[data]" torch transformers Pillow
    

## Step 1: Reading the Dataset from S3

[Imagenette](https://github.com/fastai/imagenette) is a subset of [ImageNet](https://www.image-net.org/) with 10 classes. This dataset has been hosted in a public S3 (`s3://anonymous@air-example-data-2/imagenette2/val/`). Since we are only doing inference here, we load in just the validation split.

Here, we use `ray.data.read_images` to load the validation set from S3. [Ray Data](https://docs.ray.io/en/latest/data/data.html) also supports reading from a variety of other datasources and formats.
    
    
    import ray
    
    # Disable progress bars and verbose logs
    context = ray.data.DataContext.get_current()
    context.enable_progress_bars = False
    context.verbose = False
    
    s3_uri = "s3://anonymous@air-example-data-2/imagenette2/val/"
    
    ds = ray.data.read_images(
        s3_uri, mode="RGB"
    )
    ds
    
    
    
    2025-02-05 15:55:02,527	INFO worker.py:1841 -- Started a local Ray instance.
    2025-02-05 15:55:03,276	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_15-55-01_937163_54751/logs/ray-data
    2025-02-05 15:55:03,276	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadImage]
    2025-02-05 15:55:04,483	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_15-55-01_937163_54751/logs/ray-data
    2025-02-05 15:55:04,484	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadImage]
    

Inspecting the schema, we can see that there is a column “image” in the dataset that contains the images stored as Numpy arrays.
    
    
    ds.schema()
    
    
    
    Column  Type
    ------  ----
    image   numpy.ndarray(ndim=3, dtype=uint8)
    

## Step 2: Inference on a single batch

Next, we can do inference on a single batch of data, using a pre-trained Vision Transformer from Huggingface following [this Huggingface example](https://huggingface.co/docs/transformers/tasks/image_classification#inference).

Let’s get a batch of 10 from our dataset. The batch is a dictionary for column names to data, here we have one column “image”. Each of the 10 images in the batch is represented as a Numpy array.
    
    
    single_batch = ds.take_batch(10)
    print(f"Num columns: {len(single_batch['image'])}")
    print(f"Image shape: {single_batch['image'][0].shape}")
    
    
    
    2025-02-05 15:55:37,679	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_15-55-01_937163_54751/logs/ray-data
    2025-02-05 15:55:37,679	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadImage] -> LimitOperator[limit=10]
    
    
    
    Num columns: 10
    Image shape: (480, 590, 3)
    

We can visualize the first image from this batch using PIL.
    
    
    from PIL import Image
    
    img = Image.fromarray(single_batch["image"][0])
    img
    



Now, let’s create a Huggingface Image Classification pipeline from a pre-trained Vision Transformer model.

We specify the following configurations:

  1. Set the device to “cuda” to use an NVIDIA GPU for inference.

  2. We set the batch size to 10 so that we can do inference on the entire batch at once.

We also convert the Numpy arrays representing images into PIL Images since that’s what Huggingface expects.

From the results, we see that all of the images in the batch are correctly being classified as [“tench”](https://en.wikipedia.org/wiki/Tench) which is a type of fish.
    
    
    import torch
    from transformers import pipeline
    from PIL import Image
    
    # Note, you must have GPUs on your head node in order to do this with GPUs.
    # If doing CPU inference, set DEVICE="cpu" instead.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=DEVICE)
    outputs = classifier([Image.fromarray(image_array) for image_array in single_batch["image"]], top_k=1, batch_size=10)
    del classifier # Delete the classifier to free up GPU memory.
    outputs
    
    
    
    Device set to use cuda
    
    
    
    [[{'label': 'tench, Tinca tinca', 'score': 0.9997251629829407}],
     [{'label': 'tench, Tinca tinca', 'score': 0.5197089314460754}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9994671940803528}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9946863651275635}],
     [{'label': 'tench, Tinca tinca', 'score': 0.999672532081604}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9996077418327332}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9995598196983337}],
     [{'label': 'tench, Tinca tinca', 'score': 0.991676926612854}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9948246479034424}],
     [{'label': 'tench, Tinca tinca', 'score': 0.9366462230682373}]]
    

## Step 3: Scaling up to the full Dataset with Ray Data

By using Ray Data, we can apply the same logic in the previous section to scale up to the entire dataset, leveraging all the GPUs in our cluster.

There are a couple unique properties about the inference step:

  1. Model initialization is usually pretty expensive

  2. We want to do inference in batches to maximize GPU utilization.

To address 1, we package the inference code in a `ImageClassifier` class. Using a class allows us to put the expensive pipeline loading and initialization code in the `__init__` constructor, which will run only once. The actual model inference logic is in the `__call__` method, which will be called for each batch.

To address 2, we do our inference in batches, specifying a `batch_size` to the Huggingface Pipeline. The `__call__` method takes a batch of data items, instead of a single one. As above, the batch is a dict that has one key named “image”, and the value is a Numpy array of images represented in `np.ndarray` format. Since this is the same format in step 2, and we can reuse the same inferencing logic from before.
    
    
    from typing import Dict
    import numpy as np
    
    from transformers import pipeline
    from PIL import Image
    
    # Pick the largest batch size that can fit on our GPUs.
    # If doing CPU inference you might need to lower considerably (e.g. to 10).
    BATCH_SIZE = 1024
    
    class ImageClassifier:
        def __init__(self):
            self.classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=DEVICE)
    
        def __call__(self, batch: Dict[str, np.ndarray]):
            # Convert the numpy array of images into a list of PIL images which is the format the HF pipeline expects.
            outputs = self.classifier(
                [Image.fromarray(image_array) for image_array in batch["image"]], 
                top_k=1, 
                batch_size=BATCH_SIZE)
            
            # `outputs` is a list of length-one lists. For example:
            # [[{'score': '...', 'label': '...'}], ..., [{'score': '...', 'label': '...'}]]
            batch["score"] = [output[0]["score"] for output in outputs]
            batch["label"] = [output[0]["label"] for output in outputs]
            # note: we keep the original image column in the result so that we can display the images later
            return batch
    

Then we use the `map_batches` API to apply the model to the whole dataset.

The first parameter of `map_batches` is the user-defined function (UDF), which can either be a function or a class. Here we are using a class, so the UDFs run as long-running [Ray actors](https://docs.ray.io/en/latest/ray-core/key-concepts.html#actors). For class-based UDFs, use the `concurrency` argument to specify the number of concurrent actors. The `batch_size` argument indicates the number of images in each batch.

The `num_gpus` argument specifies the number of GPUs needed for each `ImageClassifier` instance. In this case, we want 1 GPU for each model replica.

Note that `map_batches` is a lazy operation, so nothing will be computed until the dataset is consumed (see [Consuming data](https://docs.ray.io/en/latest/data/quickstart.html#consuming-data)).
    
    
    predictions = ds.map_batches(
        ImageClassifier,
        compute=ray.data.ActorPoolStrategy(size=4), # Use 4 model replicas. Change this number based on the number of GPUs in your cluster.
        num_gpus=1 if torch.cuda.is_available() else 0,  # Specify GPUs per model replica (use 0 for CPU inference)
        batch_size=BATCH_SIZE # Use batch size from above.
    )
    

### Verify and Save Results

Let’s take a small batch and verify the results. This will trigger the lazy operation on the first 5 items.
    
    
    prediction_batch = predictions.take_batch(5)
    
    
    
    2025-02-05 15:55:51,527	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_15-55-01_937163_54751/logs/ray-data
    2025-02-05 15:55:51,528	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> ActorPoolMapOperator[ReadImage->MapBatches(ImageClassifier)] -> LimitOperator[limit=5]
    2025-02-05 15:55:57,603	WARNING actor_pool_map_operator.py:280 -- To ensure full parallelization across an actor pool of size 4, the Dataset should consist of at least 4 distinct blocks. Consider increasing the parallelism when creating the Dataset.
    
    
    
    (_MapWorker pid=54998) Device set to use cuda
    

We see that all 5 of the images are correctly classified as “tench”, which is a type of fish. (You may need to scroll to see all of the samples below.)
    
    
    from PIL import Image
    from IPython.display import display
    
    
    img_count = 0
    for image, prediction in zip(prediction_batch["image"], prediction_batch["label"]):
        print("Label: ", prediction)
        print("Image:")
        # Use Jupyter to display the image inline.
        img = Image.fromarray(image)
        display(img)
        img_count += 1
    print(f"Successfully displayed {img_count} images.")
    
    
    
    Label:  tench, Tinca tinca
    Image:
    


    
    
    Label:  tench, Tinca tinca
    Image:
    


    
    
    Label:  tench, Tinca tinca
    Image:
    


    
    
    Label:  tench, Tinca tinca
    Image:
    


    
    
    Label:  tench, Tinca tinca
    Image:
    


    
    
    Successfully displayed 5 images.
    

If the samples look good, we can proceed with saving the results to an external storage, e.g., S3 or local disks. See [Ray Data Input/Output](https://docs.ray.io/en/latest/data/api/input_output.html) for all supported storages and file formats.