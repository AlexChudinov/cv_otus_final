import logging
from functools import partial

import torchvision
import tritonclient.grpc as grpcclient
import numpy as np
from PIL.ImageFile import ImageFile

from .utils import create_logger

class DINOV2Client:
    def __init__(self,
                 *,
                 model_name: str = "dinov2",
                 server_url: str = "triton:8001",
                 input_name: str = "input",
                 output_name: str = "pooler_output",
                 logger: logging.Logger = None):
        self.logger = logger or create_logger()
        self.client = grpcclient.InferenceServerClient(url=server_url)
        self.input_name = input_name
        self.outputs = [ grpcclient.InferRequestedOutput(output_name) ]
        self.infer_request = partial(self.client.async_infer, model_name=model_name, outputs=self.outputs)

    def infer_callback(self, infer_callback, result: grpcclient.InferResult, error: grpcclient.InferenceServerException | None = None):
        if error:
            self.logger.error(error)
            infer_callback(None)
        else:
            infer_callback(result.as_numpy(self.outputs[0].name()).squeeze(0))

    def image_preprocessing(self, image: ImageFile) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image_tensor = torchvision.transforms.ToTensor()(image)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_tensor = torchvision.transforms.Normalize(mean, std)(image_tensor)
        return image_tensor.to("cpu").numpy()[np.newaxis, ...]

    def __call__(self, image: ImageFile, infer_callback):
        inputs = [grpcclient.InferInput(self.input_name, (1, 3, 224, 224), "FP32")]
        inputs[0].set_data_from_numpy(self.image_preprocessing(image))
        return self.infer_request(callback=partial(self.infer_callback, infer_callback=infer_callback), inputs=inputs)
