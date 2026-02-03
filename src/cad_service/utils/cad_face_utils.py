from typing import List

from OCC.Core.TopoDS import TopoDS_Face

from cad_service.models.cad_graph.cad_graph import CADGraph
from cad_service.models.cad_graph.face_graph import FaceGraph


class CADFaceUtils:

    def __prune_faces(
        self, graph: FaceGraph, whitelist_faces: List[TopoDS_Face], root: TopoDS_Face, faces: List[TopoDS_Face]
    ):
        faces.append(root)

        face_idx = graph.get_vertex_idx_by_face(root)
        adjacency_list = graph.edges[face_idx]
        for idx in adjacency_list.nonzero()[0]:
            adj_face = graph.vertices[idx].face
            if adj_face not in faces and adj_face in whitelist_faces:
                # has same machining feature and not yet traversed
                self.__prune_faces(graph=graph, whitelist_faces=whitelist_faces, root=adj_face, faces=faces)

    def prune_non_adj_faces(self, graph: CADGraph, faces: List[TopoDS_Face], root: TopoDS_Face) -> List[TopoDS_Face]:
        """filter `faces` to include only faces that are connected to `root` face

        Will start from `root` face and recursively add faces to result if they are neighboring. Uses DFS.

        :param faces: faces
        :return: pruned faces
        """
        sel_faces = []
        self.__prune_faces(graph=graph.face_graph, faces=sel_faces, root=root, whitelist_faces=faces)

        return sel_faces
