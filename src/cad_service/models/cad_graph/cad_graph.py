import numpy as np
from OCC.Core.TopoDS import TopoDS_Shape

from cad_service.models.cad_graph.face_graph import FaceGraph, FaceGraphVertex
from cad_service.utils.hierarchical_cadnet_utils import (
    get_brep_information,
    get_face_adj,
    get_face_features,
    normalize_data,
    normalize_surface_labels,
    triangulate_shape,
)


def get_graph(work_faces, work_face_edges):
    V_1 = get_face_features(work_faces)
    _, E_1, E_2, E_3 = get_face_adj(work_face_edges, work_faces)

    surface_labels = V_1[:, -1].reshape(-1, 1)
    V_1 = V_1[:, :-1]

    V_1 = normalize_data(V_1)
    surface_labels = normalize_surface_labels(surface_labels)

    V_1 = np.concatenate((V_1, surface_labels), axis=1)

    return V_1, E_1, E_2, E_3


class CADGraph:

    def __init__(self, shape: TopoDS_Shape) -> None:
        triangulate_shape(shape)
        work_faces, work_edges, _ = get_brep_information(shape)
        V_1, E_1, E_2, E_3 = get_graph(work_faces, work_edges)

        ## face graph: topology of cad model
        # V_1: list of nodes, each has normalized feature [surface_area, centroid_x, centroid_y, centroid_z, face_type]
        # E_1: list of convex edges
        # E_2: list of concave edges
        # E_3: list of smooth or other edges
        self.face_graph = FaceGraph(
            vertices=[FaceGraphVertex(face=f.face, feature=feat) for f, feat in zip(work_faces.values(), V_1)],
            convex_edges=E_1,
            concave_edges=E_2,
            other_edges=E_3,
        )

        ## facet graph: geometry of cad model
        # V_2: list of nodes, each has normalized feature [normal_x, normal_y, normal_z, d_coefficient]
        # A_2: facets edges
        # self.facet_graph = FacetGraph(vertices=V_2, edges=A_2)

        ## connection between face graph and facet graph
        # A_3: projection from facets to faces
        # self.facet_face_connection = A_3
