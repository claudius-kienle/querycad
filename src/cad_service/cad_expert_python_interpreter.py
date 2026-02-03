import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_utils.communication_utils.models.cad_part import CADFace, CADFaceTypeCylinderMeta, CADPart
from llm_utils.communication_utils.models.view_orientation import ViewOrientation
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer

from cad_service.cad_machining_visualization_pyvista import CADMachiningVisualizationPyVista
from cad_service.img_seg.image_cad_segmentation_esemble import ImageCADSegmentationEnsemble
from cad_service.kernel.cad_loader import CADLoader
from cad_service.models.cad_part_loader.cad_part_factory import BoundingBoxFactory, CADFaceFactory


@dataclass
class Face:
    type: str
    radius: Optional[float]
    center: Tuple[float, float, float]
    extents: Tuple[float, float, float]

    @property
    def half_extents(self):
        return [x / 2 for x in self.extents]

    @staticmethod
    def from_cad_face(face: CADFace) -> "Face":
        return Face(
            type=face.face_type.type.name.lower(),
            radius=face.face_type.radius if isinstance(face.face_type, CADFaceTypeCylinderMeta) else None,
            center=tuple(face.bbox.center),
            extents=tuple(face.bbox.extents),
        )


@dataclass
class Shape:
    center: Tuple[float, float, float]
    extents: Tuple[float, float, float]
    shape: TopoDS_Shape
    faces: List[Face]

    @property
    def half_extents(self):
        return [x / 2 for x in self.extents]

    @staticmethod
    def from_cad_shape(shape: TopoDS_Shape) -> "Shape":
        bbox = BoundingBoxFactory.from_occ_shape(shape=shape)
        faces = TopologyExplorer(shape).faces()
        faces = [Face.from_cad_face(CADFaceFactory.from_occ_face(face, [0, 1, 0])) for face in faces]
        return Shape(center=bbox.center, extents=bbox.extents, shape=shape, faces=faces)


@dataclass
class Part:
    cad_part: CADPart
    center: Tuple[float, float, float]
    extents: Tuple[float, float, float]
    sides: List[str]
    faces: List[Face]

    @property
    def half_extents(self):
        return [x / 2 for x in self.extents]

    @staticmethod
    def from_cad_part(part: CADPart) -> "Part":
        return Part(
            cad_part=part,
            center=part.bounding_box.center,
            extents=part.bounding_box.extents,
            sides=list(map(lambda p: p.name.lower(), part.side)),
            faces=list(map(Face.from_cad_face, part.cad_faces)),
        )


class CADExpertPythonInterpreter:

    def __init__(
        self,
        image_seg: ImageCADSegmentationEnsemble,
        shape: TopoDS_Shape,
        visualize: bool = False,
    ) -> None:
        self.shape = shape
        self.img_seg = image_seg
        self.visualize = visualize

    def _get_parts_by_type(self, shape: Shape, instruction: str, sides: Optional[List[str]] = None) -> List[Part]:
        if sides is not None:
            v_sides = [ViewOrientation[s.upper()] for s in sides]
        else:
            v_sides = None

        parts = self.img_seg.select_parts_multi_view_advanced(shape=shape.shape, query=instruction, directions=v_sides)

        # TODO: hacky
        # see `cad_service_api.py` for counterpart
        self.img_seg.parts = parts

        assert parts is not None

        if self.visualize:
            viz = CADMachiningVisualizationPyVista()
            colors = viz.visualize_parts(shape=shape.shape, parts=parts)
            CADLoader().save_shape(shape=shape.shape, step_file_path=Path("out/output.step"), colors=colors)
            img = viz.screenshot_parts(
                shape.shape, parts, orientation=ViewOrientation.ISO, fallback_color=(0.5, 0.5, 0.5)
            )
            img.save("out/result.png")

        parts = [Part.from_cad_part(part=p) for p in parts]

        return parts

    def exec(self, code_sections: List[str], extra_globals: Optional[Dict[str, Any]] = None):
        """executes `code_sections` one by one"""
        if extra_globals is None:
            extra_globals_mapped = {}
        else:
            extra_globals_mapped = {}
            for k, v in extra_globals.items():
                if isinstance(v, CADPart):
                    extra_globals_mapped[k] = Part.from_cad_part(v)
                else:
                    extra_globals_mapped[k] = v

        restricted_globals = {
            # **globals(),
            # "get_part_by_color": self._get_part_by_color_and_parts,
            "get_parts_by_instruction": self._get_parts_by_type,
            "math": math,
            "shape": Shape.from_cad_shape(self.shape),
            "List": List,
            "Part": Part,
            "Shape": Shape,
            **extra_globals_mapped,
        }
        # restricted_locals = {}

        for code in code_sections:
            print(code)
            func = compile(code, "<string>", "exec")
            exec(func, restricted_globals, restricted_globals)
        response = restricted_globals["solution"]

        if isinstance(response, Part):
            response = response.cad_part
        elif isinstance(response, list):
            response = [r.cad_part if isinstance(r, Part) else r for r in response]
        elif isinstance(response, dict):
            raise NotImplementedError()
        else:
            pass

        return response
