import logging
from functools import partial

import tritonclient.grpc as grpcclient
import numpy as np
from PIL.ImageDraw import Draw
from PIL.ImageFile import ImageFile

from .constants import COCO_LABELS, YOLO_MODEL_SIZE
from .processor import YOLOv8Processor
from .utils import rgb_color_from_id

class YOLOClient:
    def __init__(self,
                 *,
                 model_name: str = "yolo",
                 server_url: str = "triton:8001",
                 input_name: str = "images",
                 output_name: str = "output0",
                 threshold: float = 0.1,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.client = grpcclient.InferenceServerClient(url=server_url)
        self.input_name = input_name
        self.outputs = [ grpcclient.InferRequestedOutput(output_name) ]
        self.infer_request = partial(self.client.async_infer, model_name=model_name, outputs=self.outputs)
        self.threshold = threshold

    def infer_callback(self,
                       infer_callback,
                       processor: YOLOv8Processor,
                       result: grpcclient.InferResult,
                       error: grpcclient.InferenceServerException | None = None):
        if error:
            self.logger.error(error)
            infer_callback(None)
        else:
            infer_callback(processor.postprocess(result.as_numpy(self.outputs[0].name()).squeeze()))

    def image_preprocessing(self, image: ImageFile) -> tuple[np.ndarray, YOLOv8Processor]:
        image = image.convert("RGB")
        w, h = image.size
        processor = YOLOv8Processor(YOLO_MODEL_SIZE, YOLO_MODEL_SIZE, w, h)
        return processor.preprocess(image)[None, ...].numpy().astype(np.float32), processor

    def draw_bboxes(self, output: np.ndarray, image: ImageFile) -> ImageFile:
        boxes = output[:, :4]
        scores = output[:, 4:].max(axis=-1)
        class_ids = output[:, 4:].argmax(axis=-1)
        scores_mask = scores > self.threshold
        boxes = boxes[scores_mask, ...]
        scores = scores[scores_mask]
        class_ids = class_ids[scores_mask]
        draw = Draw(image)

        for box, score, class_id in zip(boxes, scores, class_ids):
            box = box.round().astype(int)
            x, y, w, h = box
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            draw.rectangle(((x1, y1), (x2, y2)), outline=rgb_color_from_id(class_id), width=2)
            draw.text((x1, y1), text=f"{COCO_LABELS[class_id]}: {score:.2f}", fill=rgb_color_from_id(class_id))

        return image

    def __call__(self, image: ImageFile, infer_callback) -> ImageFile:
        preprocessed_image, processor = self.image_preprocessing(image)
        inputs = [grpcclient.InferInput(self.input_name, preprocessed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(preprocessed_image)
        return self.infer_request(callback=partial(self.infer_callback, infer_callback=infer_callback, processor=processor),
                                  inputs=inputs)
