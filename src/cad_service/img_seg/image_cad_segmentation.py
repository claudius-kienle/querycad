import logging
import time
from typing import List, Optional

import numpy as np
import torch
from llm_utils.communication_utils.models.view_orientation import ViewOrientation
from llm_utils.communication_utils.utils.color_utils import get_unique_colors
from OCC.Core.TopoDS import TopoDS_Shape

from cad_service.cad_machining_visualization_pyvista import CADMachiningVisualizationPyVista
from cad_service.img_seg.image_cad_ray_casting import ImageCADRayCasting
from cad_service.img_seg.image_cad_ray_casting_base import ImageCADRayCastingBase
from cad_service.img_seg.image_processing_client import ImageProcessingClient
from cad_service.img_seg.image_processing_response import ImageProcessingResponse
from cad_service.img_seg.image_segmentation_class import FaceSegmentationResult, PartSegmentationResult
from cad_service.img_seg.masked_part import MaskedPart
from cad_service.img_seg.view_orientation_extended import ViewOrientationExtended
from cad_service.models.cad_graph.cad_graph import CADGraph
from cad_service.models.cad_part_loader.cad_part_factory import CADPartFactory
from cad_service.utils.cad_face_utils import CADFaceUtils


class ImageCADSegmentation:
    """class to segment cad part by prompt"""

    def __init__(
        self,
        *,
        visualize: bool = False,
        image_processing_client: ImageProcessingClient,
        image_size: tuple = (640, 480),
        ray_casting_downscale_factor: int = 8,
        image_ray_casting: Optional[ImageCADRayCastingBase] = None,
    ):
        self.visualize = visualize
        self.client = image_processing_client
        self.visualization = CADMachiningVisualizationPyVista(image_size=image_size)
        if image_ray_casting is None:
            self.image_ray_casting = ImageCADRayCasting(image_size=image_size)
        else:
            self.image_ray_casting = image_ray_casting

    @staticmethod
    def map_faces_to_part(
        graph: CADGraph, faces_result: List[FaceSegmentationResult], sides: List[ViewOrientation], prune: bool = True
    ) -> Optional[PartSegmentationResult]:
        """combine faces segmentation result to part segmentation result

        all faces that were masked are combined to one part segmentation

        :param faces_result: faces segmentation
        :return: part segmentation
        """
        sel_faces = []
        total_mask_precision = 0
        part_size = 0

        # result with ascending distance
        faces_result = list(sorted(faces_result, key=lambda r: r.viewport_distance))

        for face in faces_result:
            # select face if at least 5% of face was selected
            if face.recall <= 0.05:
                continue

            part_size += face.size
            total_mask_precision += face.precision
            sel_faces.append(face.object)

        if len(sel_faces) == 0:
            return None

        if prune:
            sel_faces = CADFaceUtils().prune_non_adj_faces(graph=graph, faces=sel_faces, root=sel_faces[0])

        part_recall = len(sel_faces) / len(faces_result)

        cad_part = CADPartFactory.from_occ_faces(faces=sel_faces, sides=sides)
        seg_result = PartSegmentationResult(
            object=cad_part,
            size=part_size,
            precision=total_mask_precision,
            recall=part_recall,
        )
        return seg_result

    @staticmethod
    def _filter_masks_by_overlap(part_seg_result: ImageProcessingResponse):
        if len(part_seg_result.masks) == 0:
            # nothing to do
            return part_seg_result

        np_masks = [np.asanyarray(mask).astype(bool) for mask in part_seg_result.masks]
        raw_masks = part_seg_result.masks
        bounding_boxes = part_seg_result.bounding_boxes
        logits = part_seg_result.logits

        filtered_masks = []
        filtered_bboxes = []
        filtered_logits = []

        # sort by mask size ascending
        masks_size = [np.asanyarray(mask).sum() for mask in np_masks]
        np_masks = [m for _, m in sorted(zip(masks_size, np_masks), key=lambda e: e[0])]
        raw_masks = [m for _, m in sorted(zip(masks_size, raw_masks), key=lambda e: e[0])]
        bounding_boxes = [m for _, m in sorted(zip(masks_size, bounding_boxes), key=lambda e: e[0])]
        logits = [m for _, m in sorted(zip(masks_size, logits), key=lambda e: e[0])]

        np_masks_or = np.zeros_like(np_masks, dtype=bool)
        for i, (raw_mask, np_mask, bbox, logit) in enumerate(zip(raw_masks, np_masks, bounding_boxes, logits)):
            percent_overlap = (np_mask & np_masks_or).sum() / np_mask.sum()

            if percent_overlap > 0.5:
                print("mask %d overlaps with previous masks by 50 %% or more. Skipping" % i)
                continue

            np_masks_or |= np_mask

            filtered_masks.append(raw_mask)
            filtered_bboxes.append(bbox)
            filtered_logits.append(logit)

        # sort by logits descending
        filtered_masks = [m for _, m in sorted(zip(filtered_logits, filtered_masks), reverse=True)]
        filtered_bboxes = [m for _, m in sorted(zip(filtered_logits, filtered_bboxes), reverse=True)]
        filtered_logits = list(sorted(filtered_logits, reverse=True))

        return ImageProcessingResponse(masks=filtered_masks, bounding_boxes=filtered_bboxes, logits=filtered_logits)

    def _filter_masks_by_size(self, part_seg_result: ImageProcessingResponse, threshold: float = 0.45):
        frontend_pixels = (
            self.image_ray_casting.face_idx_map.numel() - self.image_ray_casting.face_id_size[0]
        )  # all - (0 -> no face -> background)

        filtered_masks = []
        filtered_bboxes = []
        filtered_logits = []
        for i, (mask, bbox, logit) in enumerate(
            zip(part_seg_result.masks, part_seg_result.bounding_boxes, part_seg_result.logits)
        ):
            mask_np = np.asanyarray(mask) == 255
            mask_pixels = np.sum(mask_np)

            ratio_frontend_pixels = mask_pixels / frontend_pixels
            if ratio_frontend_pixels > threshold:
                logging.info(
                    "mask %d has more (%.0f %%) than 50 %% frontend pixels. Skipping"
                    % (i + 1, ratio_frontend_pixels * 100)
                )
                continue

            filtered_masks.append(mask)
            filtered_bboxes.append(bbox)
            filtered_logits.append(logit)

        return ImageProcessingResponse(masks=filtered_masks, bounding_boxes=filtered_bboxes, logits=filtered_logits)

    def select_parts(
        self,
        shape: TopoDS_Shape,
        query: str,
        graph: Optional[CADGraph] = None,
        orientation: Optional[ViewOrientationExtended] = None,
        threshold: float = 0.299,
    ) -> List[MaskedPart]:
        """select parts for viewing direction that match `query`

        :param shape: occ shape
        :param parts: list of parts (ignored)
        :param query: user query
        :param visualize: whether visualize, defaults to False
        :param orientation: viewing orientation, defaults to None
        :return: list of parts selected
        """
        if orientation is None:
            orientation = ViewOrientation.ISO

        if graph is None:
            graph = CADGraph(shape=shape)

        # add shape to renderer for ray casting preprocessing
        if self.visualize:
            t1 = time.time()
        self.image_ray_casting.preprocess(shape=shape, orientation=orientation)
        if self.visualize:
            t2 = time.time()
            print("cad-seg preprocess", t2 - t1)

        if query == "":
            faces = list(self.image_ray_casting.face_idx_to_face.values())
            return [
                MaskedPart(
                    part=CADPartFactory.from_occ_faces(faces=faces, sides=[orientation]),
                    mask=None,
                    bounding_box=None,
                    logit=1,
                    part_seg_result=None,
                    faces_seg_result=[],
                )
            ]

        assert query != ""

        # variant 1: all faces randomly
        # topo = TopologyExplorer(shape)
        # shapes = list(topo.faces())
        # colors = distinctipy.get_colors(len(shapes))
        # colors = {s: c for s, c in zip(shapes, colors)}

        # variant 2: visible faces randomly
        colors = get_unique_colors(len(self.image_ray_casting.face_idx_to_face))
        colors = {f: c for f, c in zip(self.image_ray_casting.face_idx_to_face.values(), colors)}
        image = self.image_ray_casting.screenshot(shape=shape, colors=colors)

        if self.visualize:
            image.save("out/%s-01-raw.png" % orientation.name)

        # Process the image
        if self.visualize:
            t1 = time.time()
        updated_part_seg_result = self.client.process_image(image, query, threshold=threshold)
        if self.visualize:
            t2 = time.time()
            print("cad-seg SAM", t2 - t1)

        if len(updated_part_seg_result.masks) == 0:
            return []

        updated_part_seg_result = ImageCADSegmentation._filter_masks_by_overlap(part_seg_result=updated_part_seg_result)
        updated_part_seg_result = self._filter_masks_by_size(part_seg_result=updated_part_seg_result)

        # Display the results
        if self.visualize:
            self.client.display_image_with_masks(
                image, updated_part_seg_result.masks, prefix="%s-03" % orientation.name
            )
            self.client.display_image_with_boxes(
                image,
                updated_part_seg_result.bounding_boxes,
                updated_part_seg_result.logits,
                prefix="%s-03" % orientation.name,
            )

        # given segmented image, find parts
        masked_parts = []
        for idx, (mask, bbox, logit) in enumerate(
            zip(updated_part_seg_result.masks, updated_part_seg_result.bounding_boxes, updated_part_seg_result.logits)
        ):
            # prefix = "%s-04-%d" % (orientation.name, idx)
            mask = torch.from_numpy(np.copy(np.asanyarray(mask))) == 255

            faces_seg_result = self.image_ray_casting.map_mask_to_faces(mask_tensor=mask)

            if isinstance(orientation, ViewOrientation):
                sides = [orientation]
            elif isinstance(orientation, ViewOrientationExtended):
                sides = orientation.valid_sides
            else:
                raise NotImplementedError()
            updated_part_seg_result = ImageCADSegmentation.map_faces_to_part(
                graph=graph, faces_result=faces_seg_result, sides=sides
            )
            if updated_part_seg_result is None:
                continue

            updated_part = updated_part_seg_result.object

            if updated_part is None:
                continue

            masked_parts.append(
                MaskedPart(
                    part=updated_part,
                    mask=mask,
                    bounding_box=bbox,
                    logit=logit,
                    part_seg_result=updated_part_seg_result,
                    faces_seg_result=faces_seg_result,
                )
            )

        if self.visualize:
            self.visualization.screenshot_parts(
                shape=shape,
                parts=[p.part for p in masked_parts],
                orientation=orientation,
                fallback_color=[0.5, 0.5, 0.5],
            ).save("out/%s-05-final.png" % orientation.name)

        self.image_ray_casting.close()

        if self.visualize:
            t3 = time.time()
            print("cad-seg postprocess", t3 - t2)
        return masked_parts
