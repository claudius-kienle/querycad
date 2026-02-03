import base64
import gc
import os
import threading
import traceback
import warnings
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from rino_seg import RinoSeg


class FunctionMonitor:
    def __init__(self, timeout, callback):
        self.timeout = timeout
        self.callback = callback
        self.timer = None

    def reset_timer(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.callback)
        self.timer.start()


app = FastAPI()


class RequestData(BaseModel):
    image: str
    text_prompt: str
    threshold: float = 0.3


class ResponseData(BaseModel):
    masks: list
    bounding_boxes: list
    logits: list


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_mask(mask_np):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    return image_to_base64(mask_image)


global model
model = None


def load():
    global model
    if model is None:
        model = RinoSeg()


def unload():
    print("Unloading model to free up memory")
    global model
    model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Usage
monitor = FunctionMonitor(timeout=5, callback=unload)

global unload_model
unload_model = os.getenv("UNLOAD_MODEL") == "True"
print(unload_model)


@app.post("/process_image", response_model=ResponseData)
async def process_image(data: RequestData):
    warnings.filterwarnings("ignore")

    global model
    load()

    try:
        image_pil = base64_to_image(data.image)
        text_prompt = data.text_prompt

        _, detections = model.grounded_segmentation(image=image_pil, labels=[text_prompt], threshold=data.threshold)

        # Convert masks to numpy arrays and then to base64
        masks_base64 = [save_mask(detection.mask) for detection in detections]

        # Prepare bounding boxes and logits
        bounding_boxes = [detection.box.xyxy for detection in detections]
        logits_float = [detection.score for detection in detections]

        return JSONResponse(content={"masks": masks_base64, "bounding_boxes": bounding_boxes, "logits": logits_float})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        global unload_model
        if unload_model:
            unload()
        else:
            monitor.reset_timer()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
