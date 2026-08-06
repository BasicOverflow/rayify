# Object Detection Batch Inference with PyTorch

---

# Object Detection Batch Inference with PyTorch

[  ](https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=batch_inference_object_detection)   

This example demonstrates how to do object detection batch inference at scale with a pre-trained PyTorch model and [Ray Data](https://docs.ray.io/en/latest/data/data.html).

Here is what you’ll do:

  1. Perform object detection on a single image with a pre-trained PyTorch model.

  2. Scale the PyTorch model with Ray Data, and perform object detection batch inference on a large set of images.

  3. Verify the inference results and save them to an external storage.

  4. Learn how to use Ray Data with GPUs.

## Before You Begin

Install the following dependencies if you haven’t already.
    
    
    !pip install -q "ray[data]" torchvision
    

## Object Detection on a single Image with PyTorch

Before diving into Ray Data, let’s take a look at this [object detection example](https://pytorch.org/vision/stable/models.html#object-detection) from PyTorch’s official documentation. The example used a pre-trained model ([FasterRCNN_ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)) to do object detection inference on a single image.

First, download an image from the Internet.
    
    
    import requests
    from PIL import Image
    
    url = "https://s3-us-west-2.amazonaws.com/air-example-data/AnimalDetection/JPEGImages/2007_000063.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    display(img)
    



Second, load and intialize a pre-trained PyTorch model.
    
    
    from torchvision import transforms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval();
    

Then apply the preprocessing transforms.
    
    
    img = transforms.Compose([transforms.PILToTensor()])(img)
    preprocess = weights.transforms()
    batch = [preprocess(img)]
    

Then use the model for inference.
    
    
    prediction = model(batch)[0]
    

Lastly, visualize the result.
    
    
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import to_pil_image
    
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img,
                              boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4)
    im = to_pil_image(box.detach())
    display(im)
    



## Scaling with Ray Data

Then let’s see how to scale the previous example to a large set of images. We will use Ray Data to do batch inference in a streaming and distributed fashion, leveraging all the CPU and GPU resources in our cluster.

### Loading the Image Dataset

The dataset that we will be using is a subset of [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) that contains cats and dogs (the full dataset has 20 classes). There are 2434 images in the this dataset.

First, we use the `ray.data.read_images` API to load a prepared image dataset from S3. We can use the `schema` API to check the schema of the dataset. As we can see, it has one column named “image”, and the value is the image data represented in `np.ndarray` format.
    
    
    import ray
    
    ds = ray.data.read_images("s3://anonymous@air-example-data/AnimalDetection/JPEGImages")
    display(ds.schema())
    
    
    
    2025-02-05 14:22:50,021	INFO worker.py:1841 -- Started a local Ray instance.
    2025-02-05 14:22:50,698	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_14-22-49_425292_37149/logs/ray-data
    2025-02-05 14:22:50,698	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadImage]
    
    
    
    [dataset]: Run `pip install tqdm` to enable progress reporting.
    
    
    
    Column  Type
    ------  ----
    image   numpy.ndarray(ndim=3, dtype=uint8)
    

### Batch inference with Ray Data

As we can see from the PyTorch example, model inference consists of 2 steps: preprocessing the image and model inference.

#### Preprocessing

First let’s convert the preprocessing code to Ray Data. We’ll package the preprocessing code within a `preprocess_image` function. This function should take only one argument, which is a dict that contains a single image in the dataset, represented as a numpy array.
    
    
    import numpy as np
    import torch
    from torchvision import transforms
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                              fasterrcnn_resnet50_fpn_v2)
    from typing import Dict
    
    
    def preprocess_image(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        preprocessor = transforms.Compose(
            [transforms.ToTensor(), weights.transforms()]
        )
        return {
            "image": data["image"],
            "transformed": preprocessor(data["image"]),
        }
    

Then we use the `map` API to apply the function to the whole dataset. By using Ray Data’s map, we can scale out the preprocessing to all the resources in our Ray cluster. Note, the `map` method is lazy, it won’t perform execution until we start to consume the results.
    
    
    ds = ds.map(preprocess_image)
    

#### Model inference

Next, let’s convert the model inference part. Compared with preprocessing, model inference has 2 differences:

  1. Model loading and initialization is usually expensive.

  2. Model inference can be optimized with hardware acceleration if we process data in batches. Using larger batches improves GPU utilization and the overall runtime of the inference job.

Thus, we convert the model inference code to the following `ObjectDetectionModel` class. In this class, we put the expensive model loading and initialization code in the `__init__` constructor, which will run only once. And we put the model inference code in the `__call__` method, which will be called for each batch.

The `__call__` method takes a batch of data items, instead of a single one. In this case, the batch is also a dict that has one key named “image”, and the value is an array of images represented in `np.ndarray` format. We can also use the `take_batch` API to fetch a single batch, and inspect its internal data structure.
    
    
    single_batch = ds.take_batch(batch_size=3)
    display(single_batch)
    
    
    
    2025-02-05 14:22:51,757	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_14-22-49_425292_37149/logs/ray-data
    2025-02-05 14:22:51,757	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadImage->Map(preprocess_image)] -> LimitOperator[limit=3]
    
    
    
    {'image': array([array([[[137,  59,   0],
                    [139,  61,   0],
                    [145,  65,   2],
                    ...,
                    [141,  71,   2],
                    [140,  69,   7],
                    [138,  68,   8]],
     
                   [[135,  55,   0],
                    [138,  58,   0],
                    [143,  63,   2],
                    ...,
                    [142,  69,   1],
                    [140,  69,   5],
                    [138,  68,   6]],
     
                   [[141,  59,   1],
                    [145,  63,   3],
                    [146,  64,   6],
                    ...,
                    [143,  70,   1],
                    [141,  70,   4],
                    [139,  68,   4]],
     
                   ...,
     
                   [[223, 193, 157],
                    [219, 189, 153],
                    [188, 156, 118],
                    ...,
                    [151,  51,  15],
                    [147,  47,  11],
                    [142,  42,   6]],
     
                   [[224, 194, 158],
                    [225, 195, 159],
                    [224, 192, 154],
                    ...,
                    [148,  48,  12],
                    [145,  45,   9],
                    [139,  39,   3]],
     
                   [[227, 195, 157],
                    [236, 204, 166],
                    [215, 181, 144],
                    ...,
                    [148,  50,  13],
                    [145,  47,  10],
                    [138,  40,   3]]], shape=(375, 500, 3), dtype=uint8),
            array([[[ 78, 111, 104],
                    [ 80, 113, 104],
                    [ 83, 116, 105],
                    ...,
                    [153, 179, 192],
                    [177, 200, 216],
                    [192, 215, 231]],
     
                   [[ 70, 105, 101],
                    [ 76, 111, 105],
                    [ 73, 106,  99],
                    ...,
                    [154, 180, 193],
                    [141, 167, 182],
                    [127, 153, 168]],
     
                   [[ 83, 122, 127],
                    [ 54,  92,  95],
                    [ 72, 103, 105],
                    ...,
                    [157, 185, 197],
                    [157, 185, 199],
                    [154, 181, 198]],
     
                   ...,
     
                   [[  1, 103, 107],
                    [  1, 103, 107],
                    [  5, 102, 108],
                    ...,
                    [127,  40,  46],
                    [145,  54,  61],
                    [144,  53,  60]],
     
                   [[  1, 103, 107],
                    [  0, 102, 106],
                    [  2, 101, 106],
                    ...,
                    [139,  54,  61],
                    [134,  47,  53],
                    [134,  47,  53]],
     
                   [[  0, 102, 105],
                    [  0, 102, 106],
                    [  4, 103, 109],
                    ...,
                    [121,  47,  48],
                    [148,  68,  71],
                    [137,  52,  57]]], shape=(375, 500, 3), dtype=uint8),
            array([[[19,  1,  1],
                    [23,  5,  5],
                    [22,  2,  3],
                    ...,
                    [56, 29, 10],
                    [62, 34, 13],
                    [69, 41, 20]],
     
                   [[25,  7,  7],
                    [22,  4,  4],
                    [21,  1,  2],
                    ...,
                    [55, 27,  6],
                    [73, 42, 22],
                    [67, 36, 15]],
     
                   [[19,  3,  3],
                    [18,  2,  2],
                    [19,  1,  1],
                    ...,
                    [59, 28,  7],
                    [75, 43, 22],
                    [69, 37, 14]],
     
                   ...,
     
                   [[10,  2, 17],
                    [14, 11, 22],
                    [ 9, 12, 17],
                    ...,
                    [14, 18, 30],
                    [10, 12, 27],
                    [ 8, 10, 23]],
     
                   [[12,  4, 19],
                    [ 9,  6, 17],
                    [ 7, 12, 16],
                    ...,
                    [ 8, 12, 24],
                    [ 3,  7, 19],
                    [ 5,  9, 21]],
     
                   [[ 9,  2, 18],
                    [ 4,  2, 15],
                    [ 5, 10, 14],
                    ...,
                    [ 8, 10, 22],
                    [ 2,  4, 16],
                    [ 7,  9, 21]]], shape=(375, 500, 3), dtype=uint8)],
           dtype=object),
     'transformed': array([array([[[0.5372549 , 0.54509807, 0.5686275 , ..., 0.5529412 ,
                     0.54901963, 0.5411765 ],
                    [0.5294118 , 0.5411765 , 0.56078434, ..., 0.5568628 ,
                     0.54901963, 0.5411765 ],
                    [0.5529412 , 0.5686275 , 0.57254905, ..., 0.56078434,
                     0.5529412 , 0.54509807],
                    ...,
                    [0.8745098 , 0.85882354, 0.7372549 , ..., 0.5921569 ,
                     0.5764706 , 0.5568628 ],
                    [0.8784314 , 0.88235295, 0.8784314 , ..., 0.5803922 ,
                     0.5686275 , 0.54509807],
                    [0.8901961 , 0.9254902 , 0.84313726, ..., 0.5803922 ,
                     0.5686275 , 0.5411765 ]],
     
                   [[0.23137255, 0.23921569, 0.25490198, ..., 0.2784314 ,
                     0.27058825, 0.26666668],
                    [0.21568628, 0.22745098, 0.24705882, ..., 0.27058825,
                     0.27058825, 0.26666668],
                    [0.23137255, 0.24705882, 0.2509804 , ..., 0.27450982,
                     0.27450982, 0.26666668],
                    ...,
                    [0.75686276, 0.7411765 , 0.6117647 , ..., 0.2       ,
                     0.18431373, 0.16470589],
                    [0.7607843 , 0.7647059 , 0.7529412 , ..., 0.1882353 ,
                     0.1764706 , 0.15294118],
                    [0.7647059 , 0.8       , 0.70980394, ..., 0.19607843,
                     0.18431373, 0.15686275]],
     
                   [[0.        , 0.        , 0.00784314, ..., 0.00784314,
                     0.02745098, 0.03137255],
                    [0.        , 0.        , 0.00784314, ..., 0.00392157,
                     0.01960784, 0.02352941],
                    [0.00392157, 0.01176471, 0.02352941, ..., 0.00392157,
                     0.01568628, 0.01568628],
                    ...,
                    [0.6156863 , 0.6       , 0.4627451 , ..., 0.05882353,
                     0.04313726, 0.02352941],
                    [0.61960787, 0.62352943, 0.6039216 , ..., 0.04705882,
                     0.03529412, 0.01176471],
                    [0.6156863 , 0.6509804 , 0.5647059 , ..., 0.05098039,
                     0.03921569, 0.01176471]]], shape=(3, 375, 500), dtype=float32),
            array([[[0.30588236, 0.3137255 , 0.3254902 , ..., 0.6       ,
                     0.69411767, 0.7529412 ],
                    [0.27450982, 0.29803923, 0.28627452, ..., 0.6039216 ,
                     0.5529412 , 0.49803922],
                    [0.3254902 , 0.21176471, 0.28235295, ..., 0.6156863 ,
                     0.6156863 , 0.6039216 ],
                    ...,
                    [0.00392157, 0.00392157, 0.01960784, ..., 0.49803922,
                     0.5686275 , 0.5647059 ],
                    [0.00392157, 0.        , 0.00784314, ..., 0.54509807,
                     0.5254902 , 0.5254902 ],
                    [0.        , 0.        , 0.01568628, ..., 0.4745098 ,
                     0.5803922 , 0.5372549 ]],
     
                   [[0.43529412, 0.44313726, 0.45490196, ..., 0.7019608 ,
                     0.78431374, 0.84313726],
                    [0.4117647 , 0.43529412, 0.41568628, ..., 0.7058824 ,
                     0.654902  , 0.6       ],
                    [0.47843137, 0.36078432, 0.40392157, ..., 0.7254902 ,
                     0.7254902 , 0.70980394],
                    ...,
                    [0.40392157, 0.40392157, 0.4       , ..., 0.15686275,
                     0.21176471, 0.20784314],
                    [0.40392157, 0.4       , 0.39607844, ..., 0.21176471,
                     0.18431373, 0.18431373],
                    [0.4       , 0.4       , 0.40392157, ..., 0.18431373,
                     0.26666668, 0.20392157]],
     
                   [[0.40784314, 0.40784314, 0.4117647 , ..., 0.7529412 ,
                     0.84705883, 0.90588236],
                    [0.39607844, 0.4117647 , 0.3882353 , ..., 0.75686276,
                     0.7137255 , 0.65882355],
                    [0.49803922, 0.37254903, 0.4117647 , ..., 0.77254903,
                     0.78039217, 0.7764706 ],
                    ...,
                    [0.41960785, 0.41960785, 0.42352942, ..., 0.18039216,
                     0.23921569, 0.23529412],
                    [0.41960785, 0.41568628, 0.41568628, ..., 0.23921569,
                     0.20784314, 0.20784314],
                    [0.4117647 , 0.41568628, 0.42745098, ..., 0.1882353 ,
                     0.2784314 , 0.22352941]]], shape=(3, 375, 500), dtype=float32),
            array([[[0.07450981, 0.09019608, 0.08627451, ..., 0.21960784,
                     0.24313726, 0.27058825],
                    [0.09803922, 0.08627451, 0.08235294, ..., 0.21568628,
                     0.28627452, 0.2627451 ],
                    [0.07450981, 0.07058824, 0.07450981, ..., 0.23137255,
                     0.29411766, 0.27058825],
                    ...,
                    [0.03921569, 0.05490196, 0.03529412, ..., 0.05490196,
                     0.03921569, 0.03137255],
                    [0.04705882, 0.03529412, 0.02745098, ..., 0.03137255,
                     0.01176471, 0.01960784],
                    [0.03529412, 0.01568628, 0.01960784, ..., 0.03137255,
                     0.00784314, 0.02745098]],
     
                   [[0.00392157, 0.01960784, 0.00784314, ..., 0.11372549,
                     0.13333334, 0.16078432],
                    [0.02745098, 0.01568628, 0.00392157, ..., 0.10588235,
                     0.16470589, 0.14117648],
                    [0.01176471, 0.00784314, 0.00392157, ..., 0.10980392,
                     0.16862746, 0.14509805],
                    ...,
                    [0.00784314, 0.04313726, 0.04705882, ..., 0.07058824,
                     0.04705882, 0.03921569],
                    [0.01568628, 0.02352941, 0.04705882, ..., 0.04705882,
                     0.02745098, 0.03529412],
                    [0.00784314, 0.00784314, 0.03921569, ..., 0.03921569,
                     0.01568628, 0.03529412]],
     
                   [[0.00392157, 0.01960784, 0.01176471, ..., 0.03921569,
                     0.05098039, 0.07843138],
                    [0.02745098, 0.01568628, 0.00784314, ..., 0.02352941,
                     0.08627451, 0.05882353],
                    [0.01176471, 0.00784314, 0.00392157, ..., 0.02745098,
                     0.08627451, 0.05490196],
                    ...,
                    [0.06666667, 0.08627451, 0.06666667, ..., 0.11764706,
                     0.10588235, 0.09019608],
                    [0.07450981, 0.06666667, 0.0627451 , ..., 0.09411765,
                     0.07450981, 0.08235294],
                    [0.07058824, 0.05882353, 0.05490196, ..., 0.08627451,
                     0.0627451 , 0.08235294]]], shape=(3, 375, 500), dtype=float32)],
           dtype=object)}
    
    
    
    class ObjectDetectionModel:
        def __init__(self):
            # Define the model loading and initialization code in `__init__`.
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=self.weights,
                box_score_thresh=0.9,
            )
            if torch.cuda.is_available():
                # Move the model to GPU if it's available.
                self.model = self.model.cuda()
            self.model.eval()
    
        def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            # Define the per-batch inference code in `__call__`.
            batch = [torch.from_numpy(image) for image in input_batch["transformed"]]
            if torch.cuda.is_available():
                # Move the data to GPU if it's available.
                batch = [image.cuda() for image in batch]
            predictions = self.model(batch)
            return {
                "image": input_batch["image"],
                "labels": [pred["labels"].detach().cpu().numpy() for pred in predictions],
                "boxes": [pred["boxes"].detach().cpu().numpy() for pred in predictions],
            }
    

Then we use the `map_batches` API to apply the model to the whole dataset.

The first parameter of `map` and `map_batches` is the user-defined function (UDF), which can either be a function or a class. Function-based UDFs run as short-running [Ray tasks](https://docs.ray.io/en/latest/ray-core/key-concepts.html#tasks), and class-based UDFs run as long-running [Ray actors](https://docs.ray.io/en/latest/ray-core/key-concepts.html#actors). For class-based UDFs, use the `concurrency` argument to specify the number of parallel actors. The `batch_size` argument indicates the number of images in each batch.

The `num_gpus` argument specifies the number of GPUs needed for each `ObjectDetectionModel` instance. The Ray scheduler can handle heterogeous resource requirements in order to maximize the resource utilization. In this case, the `ObjectDetectionModel` instances will run on GPU and `preprocess_image` instances will run on CPU.
    
    
    ds = ds.map_batches(
        ObjectDetectionModel,
        # Use 4 model replicas. Change this number based on the number of GPUs in your cluster.
        compute=ray.data.ActorPoolStrategy(size=4),
        batch_size=4,  # Use the largest batch size that can fit in GPU memory.
        # Specify 1 GPU per model replica. Set to 0 if you are doing CPU inference.
        num_gpus=1,
    )
    

### Verify and Save Results

    
    
    from torchvision.transforms.functional import convert_image_dtype, to_tensor
    
    batch = ds.take_batch(batch_size=2)
    for image, labels, boxes in zip(batch["image"], batch["labels"], batch["boxes"]):
        image = convert_image_dtype(to_tensor(image), torch.uint8)
        labels = [weights.meta["categories"][i] for i in labels]
        boxes = torch.from_numpy(boxes)
        img = to_pil_image(draw_bounding_boxes(
            image,
            boxes,
            labels=labels,
            colors="red",
            width=4,
        ))
        display(img)
    
    
    
    2025-02-05 14:22:53,627	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-05_14-22-49_425292_37149/logs/ray-data
    2025-02-05 14:22:53,628	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> ActorPoolMapOperator[ReadImage->Map(preprocess_image)->MapBatches(ObjectDetectionModel)] -> LimitOperator[limit=2]
    2025-02-05 14:22:55,891	WARNING progress_bar.py:120 -- Truncating long operator name to 100 characters. To disable this behavior, set `ray.data.DataContext.get_current().DEFAULT_ENABLE_PROGRESS_BAR_NAME_TRUNCATION = False`.
    

 

If the samples look good, we can proceed with saving the results to an external storage, e.g., S3 or local disks. See [Ray Data Input/Output](https://docs.ray.io/en/latest/data/api/input_output.html) for all supported storage and file formats.
    
    
    ds.write_parquet("local://tmp/inference_results")