# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# from gdino import GroundingDINOAPIWrapper
import torch
from PIL import Image

# import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

# import plotly.express as px


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


class RinoSeg:

    def __init__(self, detector_id: str = None, segmenter_id: str = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"

        self.object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

        segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        self.img_processor = AutoProcessor.from_pretrained(segmenter_id)

    def detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]

    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

    def refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                mask = self.polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """

        boxes = self.get_boxes(detection_results)
        inputs = self.img_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.img_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = self.refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:

        # gdino = GroundingDINOAPIWrapper(token="de9de408622998ba83915d9564501890")
        # detections = gdino.inference(prompt=dict(image=np.asanyarray(image), prompt=labels[0]), return_mask=True)
        detections = self.detect(image, labels, threshold)
        detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections
