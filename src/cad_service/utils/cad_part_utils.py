from typing import List

from llm_utils.communication_utils.models.cad_part import CADFace, CADPart
from llm_utils.communication_utils.utils.color_utils import get_unique_colors
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer

from cad_service.models.cad_part_loader.cad_part_factory import CADFaceFactory


class CADPartUtils:
    """utility functions for cad part"""

    def add_neighboring_faces_to_part(
        self, shape: TopoDS_Shape, cad_part: CADPart, exclude_faces: List[CADFace]
    ) -> CADPart:
        """add all neighboring faces to cad part

        :param shape: occ shape
        :param cad_part: cad part
        :param exclude_faces: faces not to add even if neighboring
        :return: new cad part
        """
        topo = TopologyExplorer(shape)
        occ_faces = list(topo.faces())

        traversed_faces = []
        for face in cad_part.cad_faces:
            for occ_face in occ_faces:
                if CADFaceFactory.get_face_id(face=occ_face) == face.id:
                    # ad this face
                    if face not in traversed_faces:
                        traversed_faces.append(face)

                    # add all neighboring faces
                    edges = topo.edges_from_face(occ_face)
                    for edge in edges:
                        adj_occ_faces = topo.faces_from_edge(edge)
                        for adj_occ_face in adj_occ_faces:
                            adj_face = CADFaceFactory.from_occ_face(adj_occ_face, cad_part.color)
                            if adj_face not in traversed_faces:
                                traversed_faces.append(adj_face)
                    break

        new_faces = []
        for traversed_occ_face in traversed_faces:
            if traversed_occ_face not in cad_part.cad_faces and traversed_occ_face not in exclude_faces:
                new_faces.append(traversed_occ_face)

        all_faces = cad_part.cad_faces + new_faces

        # recolor
        all_faces = list(sorted(all_faces, key=lambda f: f.id))
        new_colors = get_unique_colors(len(all_faces))
        # print([c.id for c in all_faces])
        # print("Get %d distinct colors" % len(all_faces))
        # print(new_colors)
        all_faces = [face.copy_with(color=new_color) for face, new_color in zip(all_faces, new_colors)]

        new_part = cad_part.copy_with(cad_faces=all_faces)

        return new_part

    def add_neighboring_faces_to_parts(self, shape: TopoDS_Shape, cad_parts: List[CADPart]) -> List[CADPart]:
        """add neighboring faces to cad parts

        :param shape: occ shape
        :param cad_parts: cad parts
        :return: updated cad parts
        """
        selected_faces = []
        for cad_part in cad_parts:
            selected_faces.extend(cad_part.cad_faces)

        updated_cad_parts = [
            self.add_neighboring_faces_to_part(shape=shape, cad_part=cad_part, exclude_faces=selected_faces)
            for cad_part in cad_parts
        ]

        return updated_cad_parts
