import time
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
import torch
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from PIL import Image

from cad_service.cad_machining_visualization_pyvista import CADMachiningVisualizationPyVista
from cad_service.img_seg.image_cad_ray_casting_base import ImageCADRayCastingBase
from cad_service.img_seg.view_orientation_extended import ViewOrientationExtended
from cad_service.utils.pyvista_utils import extract_mesh_data


class ImageCADRayCasting(ImageCADRayCastingBase):
    """ray casting to couple screenshot of shape with occ shape"""

    def __init__(self, image_size: Tuple[float]):
        super().__init__(image_size)
        self.fallback_color = [0.5, 0.5, 0.5]

    def preprocess(self, shape: TopoDS_Shape, orientation: ViewOrientationExtended):
        t1 = time.time()
        mesh, face_idx_for_tri, face_idx_for_vertex, faces = extract_mesh_data(shape=shape)
        t2 = time.time()
        print("ray-cast - extract", t2 - t1)

        # Add the vertex colors
        vertex_colors = [[1.0, 0, 0] for _ in range(len(face_idx_for_vertex))]
        mesh.point_data["colors"] = vertex_colors

        # Create the plotter
        pl = pv.Plotter(off_screen=True, image_scale=1)

        # Add the mesh to the plotter with vertex colors
        pl.add_mesh(mesh, scalars="colors", rgb=True, show_edges=False, backface_culling=False)

        # Set the camera to orthographic projection
        pl.camera.ParallelProjectionOn()

        pl.window_size = self.image_size  # width, height
        vis = CADMachiningVisualizationPyVista(image_size=self.image_size)
        vis.apply_orientation(pl, orientation)
        vis.fit(pl)

        indices = np.indices((pl.window_size[1], pl.window_size[0])).reshape(2, -1).T

        # convert pixels to world coordinates
        aWidth, aHeight = pl.window_size
        nearz, farz = pl.camera.GetClippingRange()  # returns nearz, farz

        windowSize = pl.window_size
        aspect = windowSize[0] / windowSize[1]
        matrix = pl.camera.GetCompositeProjectionTransformMatrix(aspect, nearz, farz)
        np_matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])

        anXs = 2.0 * indices[:, 1] / (aWidth) - 1.0
        anYs = 2.0 * (pl.window_size[1] - indices[:, 0]) / (aHeight) - 1.0
        aZ = nearz
        aPnt = np.ones((len(anXs), 4))
        aPnt[:, 0] = anXs
        aPnt[:, 1] = anYs
        aPnt[:, 2] = aZ
        np_matrix_inv = np.linalg.inv(np_matrix)
        world_coords = np.einsum("ij,bj->bi", np_matrix_inv, aPnt)[:, :3]

        # display -> normalized display
        # normalized display -> viewport
        world_coords = np.array(world_coords)

        t3 = time.time()
        print("ray-cast - load viz", t3 - t2)
        points, rays, cells = mesh.multi_ray_trace(
            origins=world_coords,
            directions=np.array([pl.camera.direction] * len(world_coords)),
            first_point=True,
            # retry=True,
        )
        t4 = time.time()
        print("ray-cast - algo", t4 - t3)

        # compute minimal distance to camera for each face
        distance = np.linalg.norm(points - pl.camera.GetPosition(), axis=1)
        face_idx_for_rays = face_idx_for_tri[cells]
        face_distance = distance[:, None] * np.eye(face_idx_for_rays.max() + 1)[face_idx_for_rays]
        face_distance[face_distance == 0] = np.inf
        self.min_face_distance = face_distance.min(axis=0)

        # set face idx for every pixel
        # 0 -> no face, [1..] face idx
        face_idx_map = np.full([pl.window_size[1], pl.window_size[0]], fill_value=0)
        hs, ws = indices[rays].T
        face_idx_map[hs, ws] = face_idx_for_rays + 1

        self.face_idx_map = torch.as_tensor(face_idx_map)

        self.faces = faces
        self.face_idx_to_face = {idx: self.faces[idx - 1] for idx in np.unique(self.face_idx_map) if idx != 0}

        face_idx, counts = np.unique(face_idx_map, return_counts=True)
        self.face_id_size = {f.item(): c.item() for f, c in zip(face_idx, counts)}

        self.face_idx_for_tri = face_idx_for_tri
        self.face_idx_for_vertex = face_idx_for_vertex

        self.mesh = mesh
        self.pl = pl

        t5 = time.time()
        print("ray-cast - rest", t5 - t4)

    def screenshot(self, shape: TopoDS_Shape, colors: Dict[TopoDS_Face, Tuple[float, float, float]]) -> Image.Image:
        vertex_colors = [colors.get(self.faces[face_idx], self.fallback_color) for face_idx in self.face_idx_for_vertex]

        self.mesh.point_data["colors"] = vertex_colors
        self.pl.render()

        image = self.pl.screenshot()

        return Image.fromarray(image)

    def close(self):
        self.pl.close()
