import base64
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
from PIL import Image, ImageDraw

from cad_service.img_seg.image_processing_response import ImageProcessingResponse


class ImageProcessingClient:
    """client interface for image segmentation"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def base64_to_image(self, base64_str: str) -> Image.Image:
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))

    def display_image_with_masks(self, image: Image.Image, masks: List[Image.Image], prefix: Optional[str] = None):
        """display image and masks

        :param image: image
        :param masks: masks
        """
        unified_mask = np.zeros(image.size, dtype=bool).T
        for mask in masks:
            unified_mask |= np.asanyarray(mask) == 255

        masked_img = np.asanyarray(image).copy()
        masked_img[~unified_mask] = 0

        # plt.show()
        if prefix is not None:
            Image.blend(image, Image.fromarray(masked_img), alpha=0.9).save("./out/%s-masks.png" % prefix)

    def display_image_with_boxes(
        self, image: Image.Image, boxes: List[List[int]], logits: List[float], prefix: Optional[str] = None
    ):
        """display image with bounding boxes

        :param image: image
        :param boxes: bounding boxes
        :param logits: logits for bounding boxes
        """
        image_out = image.copy()

        draw = ImageDraw.Draw(image_out)

        for box, logit in zip(boxes, logits):
            x_min, y_min, x_max, y_max = box
            confidence_score = round(logit, 2)

            draw.rectangle(
                ((x_min, y_min), (x_max, y_max)), outline="red", width=4
            )  # , fill=False, edgecolor="red", linewidth=2)
            draw.text(
                (x_min + 5, y_min + 5), f"Confidence: {confidence_score}", font_size=14, fill="red"
            )  # , verticalalignment="top")

        # plt.show()
        if prefix is not None:
            image_out.save("out/%s-bbox.png" % prefix)

    def process_image(self, image: Image.Image, text_prompt: str, threshold: float = 0.249) -> ImageProcessingResponse:
        """
        Sends an image and text prompt to the FastAPI server for processing, and returns the detected masks, bounding boxes, and logits.

        Args:
            image (Image.Image): The input image to be processed.
            text_prompt (str): The text prompt to guide the object detection and segmentation.

        Returns:
            Tuple[List[Image.Image], List[List[int]], List[float]]:
                - List[Image.Image]: A list of mask images in base64 format, converted to PIL Image objects.
                - List[List[int]]: A list of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
                - List[float]: A list of confidence scores (logits) corresponding to each detected bounding box.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
            ValueError: If the response data is not in the expected format.
        """
        image_base64 = self.image_to_base64(image)
        payload = {"image": image_base64, "text_prompt": text_prompt, "threshold": threshold}

        response = requests.post(f"{self.api_url}/process_image", json=payload)
        response.raise_for_status()
        data = response.json()

        masks = [self.base64_to_image(mask_base64) for mask_base64 in data["masks"]]
        bounding_boxes = data["bounding_boxes"]
        logits = data["logits"]

        return ImageProcessingResponse(masks=masks, bounding_boxes=bounding_boxes, logits=logits)
