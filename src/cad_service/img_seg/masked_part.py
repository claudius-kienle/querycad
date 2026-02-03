from dataclasses import dataclass
from typing import Any, List, Optional

from llm_utils.communication_utils.models.cad_part import CADPart
from PIL import Image

from cad_service.img_seg.image_segmentation_class import FaceSegmentationResult, PartSegmentationResult


@dataclass
class MaskedPart:
    """masked part of occ shape"""

    part: CADPart
    mask: Image.Image
    bounding_box: dict
    logit: float
    part_seg_result: PartSegmentationResult
    faces_seg_result: List[FaceSegmentationResult]

    def copy_with(self, part: Optional[CADPart] = None) -> "MaskedPart":
        """copy with

        :param part: new part, defaults to None
        :return: new masked part
        """
        if part is None:
            part = self.part
        return MaskedPart(
            part=part,
            mask=self.mask,
            bounding_box=self.bounding_box,
            logit=self.logit,
            part_seg_result=self.part_seg_result,
            faces_seg_result=self.faces_seg_result,
        )

    def __eq__(self, value: Any) -> bool:
        if isinstance(value, MaskedPart):
            return self.part == value.part
        return False
