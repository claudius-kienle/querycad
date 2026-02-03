import numpy as np
import pyvista as pv
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer


def extract_mesh_data(shape: TopoDS_Shape):
    # Generate mesh for the shape
    BRepMesh_IncrementalMesh(shape, 0.05)  # Adjust the precision if needed

    triangles = []
    vertices = []
    face_idx_for_tri = []  # face idx for every triangle
    face_idx_for_vertex = []

    cad_faces = list(TopologyExplorer(shape).faces())

    for idx, face in enumerate(cad_faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        vertices_offset = len(vertices)

        if triangulation is not None:
            # Get the vertices
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i).Transformed(location.Transformation())
                vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                face_idx_for_vertex.append(idx)

            # Get the faces
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                triangles.append(
                    [
                        tri.Value(1) + vertices_offset - 1,
                        tri.Value(2) + vertices_offset - 1,
                        tri.Value(3) + vertices_offset - 1,
                    ]
                )
                # face_colors.append(color)
                face_idx_for_tri.append(idx)

    vertices = np.array(vertices)
    triangles = np.array(triangles)
    face_idx_for_tri = np.array(face_idx_for_tri)
    face_idx_for_vertex = np.array(face_idx_for_vertex)

    mesh = pv.PolyData(vertices, np.hstack([np.full((triangles.shape[0], 1), 3), triangles]))

    return mesh, face_idx_for_tri, face_idx_for_vertex, cad_faces
