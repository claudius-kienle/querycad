"""This script allows for the running of Hierarchical CADNet on a single CAD model and saving the final result.

# copied from hierarchical_cadnet/test_and_save_utils.py
This script requires installing pythonocc: https://github.com/tpaviot/pythonocc.
"""

import os
from collections import defaultdict

import numpy as np
from OCC.Core._BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
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
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods_Face
from OCC.Extend.DataExchange import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer, read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

EPSILON = 1e-6


class WorkFace:
    def __init__(self, index, face):
        self.index = index
        self.hash = hash(face)
        self.face = face
        self.surface_area = None
        self.centroid = None
        self.face_type = None


class WorkEdge:
    def __init__(self, index, edge):
        self.index = index
        self.hash = hash(edge)
        self.edge = edge
        self.faces = []
        self.hash_faces = []
        self.face_tags = []
        # Convex = 0, Concave = 1, Other = 2
        self.convexity = None


class WorkFacet:
    """Stores information about each facet in mesh."""

    def __init__(self, facet_tag, face_tag, node_tags):
        self.facet_tag = facet_tag
        self.face_tag = face_tag
        self.node_tags = node_tags
        self.node_coords = []
        self.normal = None
        self.d_co = None
        self.centroid = None
        self.occ_face = None
        self.occ_hash_face = None

    def get_normal(self):
        vec1 = self.node_coords[1] - self.node_coords[0]
        vec2 = self.node_coords[2] - self.node_coords[1]
        norm = np.cross(vec1, vec2)
        self.normal = norm / np.linalg.norm(norm) + EPSILON

    def get_d_coefficient(self):
        self.d_co = -(
            self.normal[0] * self.node_coords[0][0]
            + self.normal[1] * self.node_coords[0][1]
            + self.normal[2] * self.node_coords[0][2]
        )

    def get_centroid(self):
        x = (self.node_coords[0][0] + self.node_coords[1][0] + self.node_coords[1][0]) / 3
        y = (self.node_coords[0][1] + self.node_coords[1][1] + self.node_coords[1][1]) / 3
        z = (self.node_coords[0][2] + self.node_coords[1][2] + self.node_coords[1][2]) / 3

        self.centroid = [x, y, z]


def get_brep_information(shape):
    topo = TopologyExplorer(shape)
    work_faces, faces = get_faces(topo)
    work_edges = get_edges(topo, faces)

    return work_faces, work_edges, faces


def ask_point_uv2(xyz, face):
    """
    This is a general function which gives the uv coordinates from the xyz coordinates.
    The uv value is not normalised.
    """
    gpPnt = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    surface = BRep_Tool().Surface(face)

    sas = ShapeAnalysis_Surface(surface)
    gpPnt2D = sas.ValueOfUV(gpPnt, 0.01)
    uv = list(gpPnt2D.Coord())

    return uv


def ask_point_normal_face(uv, face):
    """
    Ask the normal vector of a point given the uv coordinate of the point on a face
    """
    face_ds = topods_Face(face)
    surface = BRep_Tool().Surface(face_ds)
    props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-6)

    gpDir = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        gpDir.Reverse()

    return gpDir.Coord()


def ask_edge_midpnt_tangent(edge):
    """
    Ask the midpoint of an edge and the tangent at the midpoint
    """
    result = BRep_Tool.Curve(edge)  # result[0] is the handle of curve;result[1] is the umin; result[2] is umax
    tmid = (result[1] + result[2]) / 2
    p = gp_Pnt(0, 0, 0)
    v1 = gp_Vec(0, 0, 0)
    result[0].D1(tmid, p, v1)  # handle.GetObject() gives Geom_Curve type, p:gp_Pnt, v1:gp_Vec

    return [p.Coord(), v1.Coord()]


def edge_dihedral(edge, faces):
    """
    Calculate the dihedral angle of an edge
    """
    [midPnt, tangent] = ask_edge_midpnt_tangent(edge)
    uv0 = ask_point_uv2(midPnt, faces[0])
    uv1 = ask_point_uv2(midPnt, faces[1])
    n0 = ask_point_normal_face(uv0, faces[0])
    n1 = ask_point_normal_face(uv1, faces[1])

    if edge.Orientation() == TopAbs_FORWARD:
        cp = np.cross(n0, n1)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    else:
        cp = np.cross(n1, n0)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    return s


def get_edges(topo, occ_faces):
    work_edges = {}

    edges = topo.edges()
    for edge in edges:
        faces = list(topo.faces_from_edge(edge))

        we = WorkEdge(len(work_edges), edge)

        if len(faces) > 1:
            s = edge_dihedral(edge, faces)
        else:
            s = 0

        if s == 1:
            # Convex
            edge_convexity = 0
        elif s == -1:
            # Concave
            edge_convexity = 1
        else:
            # Smooth (s==0) or other
            edge_convexity = 2

        we.convexity = edge_convexity
        we.faces = faces

        for face in faces:
            we.hash_faces.append(hash(face))
            we.face_tags.append(occ_faces.index(face))

        if len(faces) == 1:
            we.hash_faces.append(hash(faces[0]))
            we.face_tags.append(occ_faces.index(faces[0]))

        work_edges[we.hash] = we

    return work_edges


def ask_surface_area(f):
    props = GProp_GProps()

    brepgprop_SurfaceProperties(f, props)
    area = props.Mass()
    return area


def recognise_face_type(face):
    """Get surface type of B-Rep face"""
    #   BRepAdaptor to get the face surface, GetType() to get the type of geometrical surface type
    surf = BRepAdaptor_Surface(face, True)
    surf_type = surf.GetType()
    a = 0
    if surf_type == GeomAbs_Plane:
        a = 1
    elif surf_type == GeomAbs_Cylinder:
        a = 2
    elif surf_type == GeomAbs_Torus:
        a = 3
    elif surf_type == GeomAbs_Sphere:
        a = 4
    elif surf_type == GeomAbs_Cone:
        a = 5
    elif surf_type == GeomAbs_BezierSurface:
        a = 6
    elif surf_type == GeomAbs_BSplineSurface:
        a = 7
    elif surf_type == GeomAbs_SurfaceOfRevolution:
        a = 8
    elif surf_type == GeomAbs_OffsetSurface:
        a = 9
    elif surf_type == GeomAbs_SurfaceOfExtrusion:
        a = 10
    elif surf_type == GeomAbs_OtherSurface:
        a = 11

    return a


def ask_face_centroid(face):
    """Get centroid of B-Rep face."""
    mass_props = GProp_GProps()
    brepgprop.SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()

    return gPt.Coord()


def get_faces(topo):
    work_faces = {}
    faces = list(topo.faces())

    for face in faces:
        wf = WorkFace(len(work_faces), face)
        wf.face_type = recognise_face_type(face)
        wf.surface_area = ask_surface_area(face)
        wf.centroid = ask_face_centroid(face)

        work_faces[wf.hash] = wf

    return work_faces, faces


def triangulation_from_face(face, face_tag, work_facets, work_nodes, facet_face_link):
    """Triangulate a B-Rep face and get information on its facets."""
    aLoc = TopLoc_Location()
    aTriangulation = BRep_Tool().Triangulation(face, aLoc)
    aTrsf = aLoc.Transformation()

    node_link = {}

    for i in range(1, aTriangulation.NbNodes() + 1):
        node = aTriangulation.Node(i)
        node.Transform(aTrsf)
        node_tag = len(work_nodes)
        work_nodes[node_tag] = np.array([node.X(), node.Y(), node.Z()])
        node_link[i] = node_tag

    for i in range(1, aTriangulation.NbTriangles() + 1):
        node_1, node_2, node_3 = aTriangulation.Triangle(i).Get()
        node_tags = [node_link[node_1], node_link[node_2], node_link[node_3]]
        node_tags.sort()

        wf = WorkFacet(len(work_facets), face_tag, node_tags)
        facet_face_link[wf.facet_tag] = face_tag

        for node in wf.node_tags:
            wf.node_coords.append(work_nodes[node])

        wf.get_normal()
        wf.get_d_coefficient()
        wf.get_centroid()
        work_facets[wf.facet_tag] = wf

    return work_facets, work_nodes, facet_face_link


def group_nodes(work_nodes):
    new_node_link = {}
    node_groups = defaultdict(list)
    for key, val in sorted(work_nodes.items()):
        node_groups[tuple(val)].append(key)

    for nodes in node_groups.values():
        new_node_link[nodes[0]] = nodes[0]

        for i in range(1, len(nodes)):
            new_node_link[nodes[i]] = nodes[0]

    return new_node_link


def replace_nodes_of_facets(work_facets, node_link):
    for facet in work_facets.values():
        for i in range(len(facet.node_tags)):
            facet.node_tags[i] = node_link[facet.node_tags[i]]

    return work_facets


def get_edge_dicts(facets):
    edge_dict = {}
    edge_facet_dict = {}

    for facet in facets.values():
        edge_1 = tuple(sorted((facet.node_tags[0], facet.node_tags[1])))
        edge_2 = tuple(sorted((facet.node_tags[0], facet.node_tags[2])))
        edge_3 = tuple(sorted((facet.node_tags[1], facet.node_tags[2])))

        edge_1_tag = len(edge_dict)
        edge_2_tag = edge_1_tag + 1
        edge_3_tag = edge_2_tag + 1

        edge_dict[edge_1_tag] = edge_1
        edge_dict[edge_2_tag] = edge_2
        edge_dict[edge_3_tag] = edge_3

        edge_facet_dict[edge_1_tag] = facet.facet_tag
        edge_facet_dict[edge_2_tag] = facet.facet_tag
        edge_facet_dict[edge_3_tag] = facet.facet_tag

    return edge_dict, edge_facet_dict


def sort_edges_to_facets(edge_dict, edges_to_facets_dict):
    new_edge_to_facets = {}

    edge_groups = defaultdict(list)
    for key, val in sorted(edge_dict.items()):
        edge_groups[val].append(key)

    for group in edge_groups.values():
        new_edge_to_facets[group[0]] = [edges_to_facets_dict[group[0]]]

        for i in range(1, len(group)):
            new_edge_to_facets[group[0]].append(edges_to_facets_dict[group[i]])

    return new_edge_to_facets


def get_face_facet_links(facets, faces):
    projection = np.zeros((len(faces), len(facets)))

    facet_indices = sorted(list(facets.keys()))

    for key, facet in facets.items():
        a = facet.face_tag
        b = facet_indices.index(key)
        projection[a, b] = 1

    embedding = np.transpose(projection)

    return embedding, projection


def get_mesh_information(shape):
    face_dict = {}
    facets_to_faces = {}
    facets = {}
    nodes = {}
    face_tag = 0

    topo = TopologyExplorer(shape)
    faces = topo.faces()

    for face in faces:
        face_dict[face_tag] = face
        try:
            facets, nodes, facets_to_faces = triangulation_from_face(face, face_tag, facets, nodes, facets_to_faces)
        except AttributeError:
            pass
        face_tag += 1

    node_link = group_nodes(nodes)
    facets = replace_nodes_of_facets(facets, node_link)
    edge_dict, edge_facet_dict = get_edge_dicts(facets)
    edge_to_facets = sort_edges_to_facets(edge_dict, edge_facet_dict)

    return facets, edge_to_facets, facets_to_faces, nodes


def get_sparse_tensor(adj_matrix, default_value=0.0):
    idx = np.where(np.not_equal(adj_matrix, default_value))
    values = adj_matrix[idx]
    shape = np.shape(adj_matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return idx, values, shape


def get_face_features(faces):
    faces_list = []

    for face_tag, face in faces.items():
        face_list = [face.surface_area, face.centroid[0], face.centroid[1], face.centroid[2], face.face_type]
        faces_list.append(face_list)

    return np.array(faces_list, dtype=np.float32)


def get_facet_features(facets):
    facets_list = []

    for facet_tag, facet in facets.items():
        facet_list = [facet.normal[0], facet.normal[1], facet.normal[2], facet.d_co]
        facets_list.append(facet_list)

    return np.array(facets_list, dtype=np.float32)


def get_face_adj(edges, faces):
    brep_adj = np.zeros((len(faces), len(faces)))
    convex_adj = np.zeros((len(faces), len(faces)))
    concave_adj = np.zeros((len(faces), len(faces)))
    other_adj = np.zeros((len(faces), len(faces)))

    for edge in edges.values():
        a = edge.face_tags[0]
        b = edge.face_tags[1]

        brep_adj[a, b] = 1
        brep_adj[b, a] = 1

        if edge.convexity == 0:
            convex_adj[a, b] = 1
            convex_adj[b, a] = 1
        elif edge.convexity == 1:
            concave_adj[a, b] = 1
            concave_adj[b, a] = 1
        elif edge.convexity == 2:
            other_adj[a, b] = 1
            other_adj[b, a] = 1

    return brep_adj, convex_adj, concave_adj, other_adj


def get_facet_adj(facets, facet_edges):
    facet_adj = np.zeros((len(facets), len(facets)))
    facet_indices = sorted(list(facets.keys()))

    for edge in facet_edges.values():
        try:
            a = facet_indices.index(edge[0])
            b = facet_indices.index(edge[1])
            facet_adj[a, b] = 1
            facet_adj[b, a] = 1
        except:
            continue

    return facet_adj


def get_face_facet_links(facets, faces):
    projection = np.zeros((len(faces), len(facets)))
    facet_indices = sorted(list(facets.keys()))

    for key, facet in facets.items():
        a = facet.face_tag
        b = facet_indices.index(key)
        projection[a, b] = 1

    return np.transpose(projection)


def normalize_data(data):
    """Normalize data."""
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    data_norm = (data - data_min) / (data_max - data_min + EPSILON)
    return data_norm


def normalize_surface_labels(data, num_surface_types=11):
    """Normalize the surface labels."""
    data_norm = data / (num_surface_types + EPSILON)
    return data_norm


def get_graph(work_faces, work_facets, work_face_edges, work_facet_edges):
    V_1 = get_face_features(work_faces)
    V_2 = get_facet_features(work_facets)
    A_1, E_1, E_2, E_3 = get_face_adj(work_face_edges, work_faces)
    A_2 = get_facet_adj(work_facets, work_facet_edges)
    A_3 = get_face_facet_links(work_facets, work_faces)

    surface_labels = V_1[:, -1].reshape(-1, 1)
    V_1 = V_1[:, :-1]

    V_1 = normalize_data(V_1)
    V_2 = normalize_data(V_2)
    surface_labels = normalize_surface_labels(surface_labels)

    V_1 = np.concatenate((V_1, surface_labels), axis=1)

    return [V_1, E_1, E_2, E_3, V_2, A_2, A_3]


def read_step_file(filename):
    """Reads STEP file."""
    if not os.path.exists(filename):
        print(filename, " not exists")
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    reader.WS().TransferReader()

    topo = TopologyExplorer(shape)

    return shape, topo


def read_step_with_labels(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, " not exists")
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = []
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = name
            id_map.append(int(nameid))

    return shape, id_map, topo


def triangulate_shape(shape, linear_deflection=0.9, angular_deflection=0.5):
    """Triangulate the shape into a faceted mesh."""
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    assert mesh.IsDone()


def create_hier_graphs(step_path, with_labels=False):
    if with_labels:
        shape, labels, topo = read_step_with_labels(step_path)
    else:
        shape, topo = read_step_file(step_path)
        labels = None

    triangulate_shape(shape)
    work_faces, work_edges, faces = get_brep_information(shape)
    facet_dict, edge_facet_link, facet_face_link, node_dict = get_mesh_information(shape)
    graph = get_graph(work_faces, facet_dict, work_edges, edge_facet_link)

    return graph, shape, labels


def write_step_wth_prediction(filename, shape, prediction):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()

    loc = TopLoc_Location()
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    counter = 0
    for face in faces:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(prediction[counter])))
        counter += 1

    writer.Write(filename)


def test_step(x):
    test_logits = model(x, training=False)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    return y_pred
