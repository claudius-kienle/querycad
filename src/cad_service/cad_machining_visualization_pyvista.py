from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from llm_utils.communication_utils.models.cad_part import CADFace, CADPart
from llm_utils.communication_utils.models.view_orientation import ViewOrientation
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from PIL import Image

from cad_service.img_seg.view_orientation_extended import ViewOrientationExtended
from cad_service.models.cad_part_loader.cad_part_factory import CADFaceFactory
from cad_service.utils.pyvista_utils import extract_mesh_data


class CADMachiningVisualizationPyVista:
    """visualization class for cad object"""

    def __init__(self, image_size: Tuple = (1920, 1080)):
        self.image_size = image_size

    def fit(self, pl: pv.Plotter, padding: float = 0):
        # # bootstrapped from https://github.com/pyvista/pyvista/blob/5f976a025980dec08e4ab0fc7d6012fe0106efb7/pyvista/plotting/camera.py#L783-L886
        pl.camera.enable_parallel_projection()

        pl.renderer.reset_camera()  # ensures that object is entirey visible in img
        # capture image and check how much percent of image size occupied by object
        # then scale the image to capture entire image size + 0.99 for padding
        image = pl.screenshot()
        indices = np.asarray(((np.asanyarray(image) != [255, 255, 255]).all(axis=2)).nonzero())

        max_h, max_w = indices.max(1)
        max_h = max_h - pl.renderer.height / 2
        max_w = max_w - pl.renderer.width / 2
        content_max = max(max_w / (pl.renderer.width / 2), max_h / (pl.renderer.height / 2))

        min_h, min_w = indices.min(1)
        min_h = pl.renderer.height / 2 - min_h
        min_w = pl.renderer.width / 2 - min_w
        content_min = max(min_w / (pl.renderer.width / 2), min_h / (pl.renderer.height / 2))

        scale = max(content_max, content_min)
        pl.camera.parallel_scale *= scale / 0.99
        pl.render()

    def apply_orientation(self, pl: pv.Plotter, orientation: ViewOrientationExtended):
        # orientation matches the 3D Viewer on Windows
        # roll, azimuth: https://docs.pyvista.org/api/core/camera.html#controlling-camera-rotation
        if isinstance(orientation, ViewOrientation):
            if orientation == ViewOrientation.TOP:
                pl.view_xy()
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.BOTTOM:
                pl.view_xy(negative=True)
                pl.camera.roll = 180
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.LEFT:
                pl.view_yz(negative=True)
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.RIGHT:
                pl.view_yz()
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.FRONT:
                pl.view_xz()
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.BACK:
                pl.view_xz(negative=True)
                # pl.camera.elevation += 1
                pl.camera.azimuth = 1
            elif orientation == ViewOrientation.ISO:
                pl.view_isometric()
            else:
                raise NotImplementedError()
        else:
            pl.render()
            pl.view_xz()
            pl.camera.roll = orientation.roll
            pl.camera.elevation += orientation.elevation
            pl.camera.azimuth += orientation.azimuth
            pl.render()

    def screenshot_shape(
        self, shape: TopoDS_Shape, orientation: ViewOrientationExtended, fallback_color: Optional[Tuple] = None
    ) -> Image.Image:
        return self.screenshot_parts(shape=shape, parts=[], orientation=orientation, fallback_color=fallback_color)

    def screenshot_parts(
        self,
        shape: TopoDS_Shape,
        parts: List[CADPart],
        orientation: ViewOrientationExtended,
        fallback_color: Optional[Tuple],
    ) -> Image.Image:
        if fallback_color is None:
            fallback_color = (1.0, 0, 0)

        colors = self.visualize_parts(shape, parts)

        return self.screenshot_faces(shape=shape, colors=colors, orientation=orientation, fallback_color=fallback_color)

    def screenshot_faces(
        self,
        shape: TopoDS_Shape,
        colors: Dict[TopoDS_Face, Tuple[float]],
        orientation: ViewOrientationExtended,
        fallback_color: Optional[Tuple],
    ) -> Image.Image:
        mesh, face_idx_for_tri, face_idx_for_vertex, faces = extract_mesh_data(shape=shape)
        vertex_colors = [colors.get(faces[face_idx], fallback_color) for face_idx in face_idx_for_vertex]

        mesh.point_data["colors"] = vertex_colors

        pl = pv.Plotter(off_screen=True, image_scale=1)

        # Add the mesh to the plotter with vertex colors
        pl.add_mesh(mesh, scalars="colors", rgb=True, show_edges=False, backface_culling=False)

        # Set the camera to orthographic projection
        pl.camera.ParallelProjectionOn()

        pl.window_size = self.image_size  # width, height
        self.apply_orientation(pl, orientation)
        self.fit(pl)

        pl.render()
        image = pl.screenshot()

        return Image.fromarray(image)

    def visualize_parts(self, shape: TopoDS_Shape, parts: List[CADPart]):
        faces = TopologyExplorer(shape).faces()
        faces_map = {CADFaceFactory.get_face_id(face): face for face in faces}
        return {faces_map[face.id]: part.color for part in parts for face in part.cad_faces}

    def visualize_faces(self, shape: TopoDS_Shape, cad_faces: List[CADFace]):
        faces = TopologyExplorer(shape).faces()
        faces_map = {CADFaceFactory.get_face_id(face): face for face in faces}
        return {faces_map[face.id]: face.color for face in cad_faces}
