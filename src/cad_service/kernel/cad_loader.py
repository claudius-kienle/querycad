from pathlib import Path
from typing import Dict

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.Quantity import Quantity_Color, Quantity_TypeOfColor
from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from OCC.Extend.TopologyUtils import TopologyExplorer


class CADLoader:

    def read_shape(self, step_file_path: Path) -> TopoDS_Shape:
        assert step_file_path.exists()

        step_reader = STEPControl_Reader()
        step_reader.ReadFile(step_file_path.as_posix())
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        return shape

    def fuse_shape_subshapes(self, shape: TopoDS_Shape) -> TopoDS_Shape:
        topo = TopologyExplorer(shape)
        shapes = list(topo.solids())
        if len(shapes) == 0:
            return shape
        last_shape = shapes[0]
        for shape in shapes[1:]:
            fuse = BRepAlgoAPI_Fuse(last_shape, shape)
            # fuse = BRepAlgo_Fuse(last_shape, shape) # legacy method, but works better than the one BRepAlgoAPI_Fuse
            fuse.SimplifyResult()
            last_shape = fuse.Shape()
        return last_shape

    def save_shape(self, shape: TopoDS_Shape, step_file_path: Path, colors: Dict[TopoDS_Face, list]):
        doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))

        # Get the shape and color tools
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

        explorer = TopologyExplorer(shape)
        for face in explorer.faces():
            if face in colors:
                color = colors[face]
                color = Quantity_Color(*color, Quantity_TypeOfColor.Quantity_TOC_RGB)
            else:
                color = Quantity_Color(0.5, 0.5, 0.5, Quantity_TypeOfColor.Quantity_TOC_RGB)
            color_tool.SetColor(shape_tool.AddShape(face, False), color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)

        step_writer = STEPCAFControl_Writer()
        step_writer.Transfer(doc, STEPControl_AsIs)
        step_writer.Write(step_file_path.as_posix())
