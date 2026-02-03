from collections import defaultdict
from typing import List, Tuple

import torch
from llm_utils.communication_utils.models.cad_part import CADPart

from cad_service.img_seg.image_segmentation_class import FaceSegmentationResult, PartSegmentationResult
from cad_service.models.cad_part_loader.cad_part_factory import CADFaceFactory


class ImageCADRayCastingBase:
    """ray casting to couple screenshot of shape with occ shape"""

    def __init__(self, image_size: Tuple[float, float]):
        assert len(image_size) == 2
        self.image_size = image_size

    def map_mask_to_faces(self, mask_tensor: torch.Tensor) -> List[FaceSegmentationResult]:
        """given mask of view, returns faces that were masked

        :param mask_tensor: boolean tensor
        :return: list of `ImageSegmentationClass`
        """
        faces_occurrences = defaultdict(int)
        assert self.face_idx_map is not None

        face_ids = self.face_idx_map[mask_tensor]
        face_ids, counts = torch.unique(face_ids, return_counts=True)
        total_occurrences = counts.sum().item()
        faces_occurrences = {f.item(): c.item() for f, c in zip(face_ids, counts)}

        result = []
        for face_id, count in faces_occurrences.items():
            if face_id not in self.face_idx_to_face:
                continue
            face = self.face_idx_to_face[face_id]
            viewport_distance = self.min_face_distance[face_id - 1]
            face_size = self.face_id_size[face_id]
            if face is None:
                # pixels with no face
                continue

            result.append(
                FaceSegmentationResult(
                    object=face,
                    size=count,
                    precision=count / total_occurrences,
                    recall=count / face_size,
                    viewport_distance=viewport_distance,
                )
            )

        return result

    def map_faces_result_to_part_result(
        self, masked_faces_result: List[FaceSegmentationResult], part: CADPart
    ) -> PartSegmentationResult:
        """computes how much of part is selected by mask

        :param masked_faces_result: faces selected by mask
        :param part: part to check
        :return: part segmentation result
        """
        part_size = 0
        part_recall = 0
        part_precision = 0
        for part_face in part.cad_faces:
            for masked_face_result in masked_faces_result:
                if part_face.id == CADFaceFactory.get_face_id(masked_face_result.object):
                    part_size += masked_face_result.size
                    part_precision += masked_face_result.precision
                    part_recall += masked_face_result.recall
                    break

        # how many percent of the part's faces where masked
        part_recall /= len(part.cad_faces)

        return PartSegmentationResult(object=part, size=part_size, precision=part_precision, recall=part_recall)

    def map_faces_result_to_parts_result(
        self, masked_faces_result: List[FaceSegmentationResult], parts: List[CADPart]
    ) -> List[PartSegmentationResult]:
        """computes how much the `parts` where selected by the `mask`

        :param masked_faces_result: mask result
        :param parts: parts to check
        :return: segmentation result for each part
        """

        parts_result = []

        for part in parts:
            part_result = self.map_faces_result_to_part_result(masked_faces_result=masked_faces_result, part=part)
            if part_result.size > 0:
                # only add part if in mask with at least one pixel
                parts_result.append(part_result)

        return parts_result
