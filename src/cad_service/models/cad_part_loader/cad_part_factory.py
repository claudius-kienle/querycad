from typing import List, Optional

import numpy as np
from llm_utils.communication_utils.models.cad_part import (
    BoundingBox,
    CADEdge,
    CADFace,
    CADFaceType,
    CADFaceTypeCylinderMeta,
    CADFaceTypeMeta,
    CADFaceTypeOtherMeta,
    CADPart,
)
from llm_utils.communication_utils.models.unique_color_supplier import Color, UniqueColorSupplier
from llm_utils.communication_utils.models.view_orientation import ViewOrientation
from llm_utils.communication_utils.utils.color_utils import get_unique_colors
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Iterator, TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer


class CADFaceTypeFactory:
    """factory to create face face from occ"""

    @staticmethod
    def recognise_face_type(face: TopoDS_Face) -> CADFaceType:
        """Get surface type of B-Rep face"""
        # BRepAdaptor to get the face surface, GetType() to get the type of geometrical surface type
        surf = BRepAdaptor_Surface(face, True)
        surf_type = surf.GetType()

        if surf_type == GeomAbs_Plane:
            return CADFaceType.PLANE
        elif surf_type == GeomAbs_Cylinder:
            return CADFaceType.CYLINDER
        elif surf_type == GeomAbs_Torus:
            return CADFaceType.TORUS
        elif surf_type == GeomAbs_Sphere:
            return CADFaceType.SPHERE
        elif surf_type == GeomAbs_Cone:
            return CADFaceType.CONE
        elif surf_type == GeomAbs_BezierSurface:
            return CADFaceType.BEZIER_SURFACE
        elif surf_type == GeomAbs_BSplineSurface:
            return CADFaceType.BSPLINE_SURFACE
        elif surf_type == GeomAbs_SurfaceOfRevolution:
            return CADFaceType.SURFACE_OF_REVOLUTION
        elif surf_type == GeomAbs_OffsetSurface:
            return CADFaceType.OFFSET_SURFACE
        elif surf_type == GeomAbs_SurfaceOfExtrusion:
            return CADFaceType.SURFACE_OF_EXTRUSION
        elif surf_type == GeomAbs_OtherSurface:
            return CADFaceType.OTHER_SURFACE


class CADFaceTypeMetaFactory:
    @staticmethod
    def from_occ(face: TopoDS_Face) -> "CADFaceTypeMeta":
        face_type = CADFaceTypeFactory.recognise_face_type(face=face)

        if face_type == CADFaceType.CYLINDER:
            surf = BRep_Tool.Surface(face)
            adaptSurf = GeomAdaptor_Surface(surf)

            cylinder = adaptSurf.Cylinder()
            radius = cylinder.Radius() / 1000  # mm to m
            # direction = [cylinder.Axis().Direction().X(), cylinder.Axis().Direction().Y(), cylinder.Axis().Direction().Z()]

            return CADFaceTypeCylinderMeta(radius=radius)
        else:
            return CADFaceTypeOtherMeta(type=face_type)


def get_edges(shape: TopoDS_Shape) -> List[TopoDS_Edge]:
    if isinstance(shape, TopoDS_Edge):
        return [shape]

    it = TopoDS_Iterator(shape)
    edges = []
    while it.More():
        subshape = it.Value()
        edges.extend(get_edges(subshape))
        it.Next()
    return edges


class BoundingBoxFactory:
    """factory to create bounding box from occ"""

    @staticmethod
    def from_occ_shape(shape: TopoDS_Shape) -> "BoundingBox":
        bbox = Bnd_Box()
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        assert mesh.IsDone()
        brepbndlib_Add(shape, bbox, True)

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        mins = np.asarray([xmin, ymin, zmin])
        maxs = np.asarray([xmax, ymax, zmax])
        mins, maxs = mins / 1000, maxs / 1000  # mm to m

        bbox = BoundingBox(min_point=mins.tolist(), max_point=maxs.tolist())
        return bbox


class CADEdgeFactory:
    """factory to create cad edge from occ"""

    @staticmethod
    def from_occ_edge(edge: TopoDS_Edge) -> "CADEdge":
        edge_props = GProp_GProps()
        brepgprop.LinearProperties(edge, edge_props)
        return CADEdge(length=edge_props.Mass(), center_of_mass=edge_props.CentreOfMass().Coord())


class CADFaceFactory:
    """factory to create cad face from occ"""

    @staticmethod
    def get_face_id(face: TopoDS_Face) -> int:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        surface_area = props.Mass()

        bbox = BoundingBoxFactory.from_occ_shape(shape=face)

        id = hash(tuple(list(map(lambda v: round(v, 5), bbox.center)) + [round(surface_area, 5)]))

        return id

    @staticmethod
    def from_occ_face(face: TopoDS_Face, color: Color) -> "CADFace":
        bbox = BoundingBoxFactory.from_occ_shape(shape=face)

        id = CADFaceFactory.get_face_id(face=face)

        face_type = CADFaceTypeMetaFactory.from_occ(face=face)
        edges = get_edges(face)
        up_edges = list(map(CADEdgeFactory.from_occ_edge, edges))

        return CADFace(bbox=bbox, id=id, face_type=face_type, edges=up_edges, color=color)


class CADPartFactory:
    """factory to create cad part from occ"""

    @staticmethod
    def from_occ_faces(faces: List[TopoDS_Face], sides: Optional[List[ViewOrientation]] = None) -> "CADPart":
        sew = BRepOffsetAPI_Sewing()
        for f in faces:
            sew.Add(f)
        sew.Perform()
        shape = sew.SewedShape()

        bbox = BoundingBoxFactory.from_occ_shape(shape=shape)

        part_color = UniqueColorSupplier.get_instance().get_color()

        if sides is None:
            sides = list(ViewOrientation)

        colors = get_unique_colors(len(faces))

        return CADPart(
            bounding_box=bbox,
            color=part_color,
            cad_faces=[CADFaceFactory.from_occ_face(face=face, color=color) for color, face in zip(colors, faces)],
            side=sides,
        )

    def from_occ_shape(self, shape: TopoDS_Shape) -> CADPart:
        faces = list(TopologyExplorer(shape).faces())
        return self.from_occ_faces(faces=faces)
