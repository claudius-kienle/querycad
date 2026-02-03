from dataclasses import dataclass
from typing import List

import numpy as np
from OCC.Core.TopoDS import TopoDS_Face


@dataclass
class FaceGraphVertex:
    face: TopoDS_Face
    feature: np.ndarray


@dataclass
class FaceGraph:
    vertices: List[FaceGraphVertex]
    convex_edges: np.ndarray
    concave_edges: np.ndarray
    other_edges: np.ndarray

    def get_vertex_idx_by_face(self, face: TopoDS_Face) -> int:
        for idx, vertex in enumerate(self.vertices):
            if vertex.face == face:
                return idx
        raise RuntimeError()

    @property
    def edges(self) -> np.ndarray:
        return self.convex_edges.astype(bool) | self.concave_edges.astype(bool) | self.other_edges.astype(bool)
