from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class ImageProcessingResponse:
    """response from image segmentation"""

    masks: List[Image.Image]
    bounding_boxes: List[List[int]]
    logits: List[float]
