from dataclasses import dataclass
from typing import Generic, TypeVar

from llm_utils.communication_utils.models.cad_part import CADPart
from OCC.Core.TopoDS import TopoDS_Face

T = TypeVar("T")


@dataclass
class ImageSegmentationClass(Generic[T]):
    object: T
    size: int
    precision: float
    recall: float


@dataclass
class FaceSegmentationResult(ImageSegmentationClass[TopoDS_Face]):
    viewport_distance: float


class PartSegmentationResult(ImageSegmentationClass[CADPart]):
    pass
