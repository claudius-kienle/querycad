import multiprocessing
from typing import Dict, List, Optional

from llm_utils.communication_utils.models.cad_part import CADPart
from llm_utils.communication_utils.models.view_orientation import ViewOrientation
from llm_utils.communication_utils.utils.color_utils import get_unique_colors
from OCC.Core.TopoDS import TopoDS_Shape

from cad_service.img_seg.image_cad_ray_casting import ImageCADRayCasting
from cad_service.img_seg.image_cad_segmentation import ImageCADSegmentation
from cad_service.img_seg.image_processing_client import ImageProcessingClient
from cad_service.img_seg.masked_part import MaskedPart
from cad_service.img_seg.view_orientation_extended import ViewOrientationExtended
from cad_service.kernel.cad_scc import Graph
from cad_service.models.cad_graph.cad_graph import CADGraph
from cad_service.models.cad_part_loader.cad_part_factory import CADFaceFactory


class ImageCADSegmentationEnsemble:
    """class to segment cad part by prompt"""

    def __init__(
        self,
        *,
        visualize: bool = False,
        image_processing_client: ImageProcessingClient,
        image_size: tuple = (640, 480),
        ray_casting_downscale_factor: int = 8,
        segmentation_threshold: float = 0.3
    ):
        self.image_size = image_size
        self.visualize = visualize
        self.image_processing_client = image_processing_client
        self.ray_casting_downscale_factor = ray_casting_downscale_factor
        self.image_ray_casting = ImageCADRayCasting(image_size=image_size)
        self.image_cad_segmentation = ImageCADSegmentation(
            visualize=visualize,
            image_processing_client=image_processing_client,
            image_size=image_size,
            ray_casting_downscale_factor=ray_casting_downscale_factor,
            image_ray_casting=self.image_ray_casting,
        )
        self.segmentation_threshold = segmentation_threshold

    def unify_masked_parts(self, parts: Dict[ViewOrientation, List[CADPart]]) -> List[CADPart]:
        """unify parts selected from different view direction but have at least one common face

        builds graph where every part is a vertex and a edge is between two parts if one face overlaps

        :param parts: parts to unify
        :return: unified parts
        """
        view_directions = list(parts.keys())

        idx = 0
        idx_map = {}
        flattened_parts: Dict[CADPart] = {}
        for mp in parts.values():
            for p in mp:
                if p.id not in idx_map:
                    flattened_parts[p.id] = p
                    # if p not in flattened_parts:
                    idx_map[p.id] = idx
                    idx += 1
                else:
                    flattened_parts[p.id] = flattened_parts[p.id].unify(p)

        n_parts = len(idx_map)
        graph = Graph(vertex=n_parts)

        for i in range(len(view_directions)):
            view_direction = view_directions[i]
            view_parts = parts[view_direction]
            for view_part in view_parts:
                for j in range(i + 1, len(view_directions)):
                    diff_view_direction = view_directions[j]
                    diff_view_parts = parts[diff_view_direction]
                    assert diff_view_direction != view_direction

                    for diff_view_part in diff_view_parts:
                        if diff_view_part.id == view_part.id:
                            # exactly the same part, nothing to do
                            continue
                        if diff_view_part.overlaps(view_part):
                            graph.add_edge(idx_map[view_part.id], idx_map[diff_view_part.id])
                            graph.add_edge(idx_map[diff_view_part.id], idx_map[view_part.id])

        # compute strongly connected components
        sccs = graph.get_sccs()

        # map to parts
        unified_parts: List[CADPart] = []
        for scc in sccs:
            unified_part_list: List[CADPart] = []
            for idx in scc:
                # get part id given vertex idx
                id = list(idx_map.keys())[list(idx_map.values()).index(idx)]

                for part in flattened_parts.values():
                    if part.id == id:
                        unified_part_list.append(part)
                        break

            assert len(scc) == len(unified_part_list)

            unified_part = unified_part_list[0]
            for next_part in unified_part_list[1:]:
                unified_part = unified_part.unify(next_part)
            unified_parts.append(unified_part)

        return unified_parts

    def select_parts_multi_view_unify(
        self, shape: TopoDS_Shape, query: str, directions: Optional[List[ViewOrientation]]
    ) -> List[CADPart]:
        """get all parts of `shape` that match `query` by multi-view segmentation"""
        masked_parts = self.select_parts_multi_view(shape=shape, query=query, directions=directions)
        res_parts = {k: [p.part for p in v] for k, v in masked_parts.items()}
        return self.unify_masked_parts(parts=res_parts)

    def _select_parts_multi_view_sequential(
        self, shape: TopoDS_Shape, query: str, graph: CADGraph, directions: List[ViewOrientation]
    ) -> Dict[ViewOrientation, List[MaskedPart]]:
        """select parts that match `query` from all viewing directions

        will run one view direction one after another

        :param shape: occ shape
        :param parts: parts, currently ignored
        :param query: user query
        :return: dict with view direction and detected parts
        """

        image_ray_casting = ImageCADRayCasting(image_size=self.image_size)
        image_cad_segmentation = ImageCADSegmentation(
            visualize=self.visualize,
            image_processing_client=self.image_processing_client,
            image_size=self.image_size,
            ray_casting_downscale_factor=self.ray_casting_downscale_factor,
            image_ray_casting=image_ray_casting,
        )

        results = {}
        for direction in directions:
            results[direction] = image_cad_segmentation.select_parts(
                shape=shape,
                query=query,
                orientation=direction,
                graph=graph,
                threshold=self.segmentation_threshold,
            )

        return results

    def _select_parts_multi_view_multiprocess(
        self, shape: TopoDS_Shape, query: str, graph: CADGraph, directions: List[ViewOrientation]
    ) -> Dict[ViewOrientation, List[MaskedPart]]:
        """select parts that match `query` from all viewing directions"""

        def process(
            shape: TopoDS_Shape,
            query: str,
            direction: ViewOrientation,
            graph: CADGraph,
            threshold: float,
            queue: multiprocessing.Queue,
        ):
            image_ray_casting = ImageCADRayCasting(image_size=self.image_size)
            image_cad_segmentation = ImageCADSegmentation(
                visualize=self.visualize,
                image_processing_client=self.image_processing_client,
                image_size=self.image_size,
                ray_casting_downscale_factor=self.ray_casting_downscale_factor,
                image_ray_casting=image_ray_casting,
            )
            res = image_cad_segmentation.select_parts(
                shape=shape, query=query, orientation=direction, graph=graph, threshold=threshold
            )
            queue.put((direction, res))

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        jobs = []
        for direction in directions:
            p = multiprocessing.Process(
                target=process, args=(shape, query, direction, graph, self.segmentation_threshold, queue)
            )
            jobs.append(p)
            p.start()

        # Ensure all processes have finished
        for job in jobs:
            job.join()

        results = {}
        while not queue.empty():
            d, e = queue.get()
            results[d] = e

        return results

    def select_parts_multi_view(
        self, shape: TopoDS_Shape, query: str, directions: Optional[List[ViewOrientation]]
    ) -> Dict[ViewOrientation, List[MaskedPart]]:
        """select parts that match `query` from all viewing directions"""
        if directions is None:
            directions = [
                ViewOrientation.TOP,
                ViewOrientation.BOTTOM,
                ViewOrientation.LEFT,
                ViewOrientation.RIGHT,
                ViewOrientation.FRONT,
                ViewOrientation.BACK,
            ]
        graph = CADGraph(shape=shape)

        results = self._select_parts_multi_view_multiprocess(
            shape=shape, query=query, graph=graph, directions=directions
        )

        len_parts = sum(len(e) for e in results.values())
        colors = iter(get_unique_colors(len_parts))
        c_results = {}
        for d, m_parts in results.items():
            c_results[d] = [
                part.copy_with(CADPart.copy_with_new_color(part=part.part, color=color))
                for part, color in zip(m_parts, colors)
            ]

        return c_results

    def select_parts_multi_view_advanced(
        self,
        shape: TopoDS_Shape,
        query: str,
        directions: Optional[List[ViewOrientation]],
        n_shots: int = 8,
    ):
        all_directions = [
            ViewOrientation.TOP,
            ViewOrientation.BOTTOM,
            ViewOrientation.LEFT,
            ViewOrientation.RIGHT,
            ViewOrientation.FRONT,
            ViewOrientation.BACK,
        ]
        if directions is None:
            directions = all_directions

        if n_shots == 6:
            # requested directions `directions` are the same as the angles used for segmentation
            return self.select_parts_multi_view_unify(shape=shape, query=query, directions=directions)

        elif n_shots == 8:
            all_directions_v8 = list(ViewOrientationExtended.get_dirs_8().values())
            res_parts = self.select_parts_multi_view_unify(shape=shape, query=query, directions=all_directions_v8)

            if directions != all_directions:
                print("filter directions")
                # we need to filter directions
                parts_for_side = {}
                for direction in directions:
                    view_parts = self.parts_view_on_side(shape=shape, parts=res_parts, side=direction)
                    parts_for_side[direction] = view_parts

                return self.unify_masked_parts(parts=parts_for_side)

            else:
                return res_parts

    def parts_view_on_side(self, shape: TopoDS_Shape, parts: List[CADPart], side: ViewOrientation):
        self.image_ray_casting.preprocess(shape=shape, orientation=side)

        face_id_size = self.image_ray_casting.face_id_size
        face_idx_to_face = self.image_ray_casting.face_idx_to_face

        face_to_size = {face_idx_to_face[face_idx]: size for face_idx, size in face_id_size.items() if face_idx != 0}

        view_parts = []
        for part in parts:
            if side not in part.side:
                # part was not detected from a viewing angle that sees `side`
                # this prevents that we look through the object and select a part from the opposite side
                continue

            size = 0
            for face in part.cad_faces:
                for view_face, view_face_size in face_to_size.items():
                    if face.id == CADFaceFactory().get_face_id(face=view_face):
                        size += view_face_size

            if size > 0:
                # part is visible from side `side`
                view_parts.append(part)

        return view_parts
