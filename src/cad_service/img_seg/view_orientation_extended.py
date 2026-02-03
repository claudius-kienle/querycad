from dataclasses import dataclass, field
from typing import Dict, List

from llm_utils.communication_utils.models.view_orientation import ViewOrientation


@dataclass
class ViewOrientationExtended:
    # in degrees
    elevation: int
    azimuth: int
    valid_sides: List[ViewOrientation]
    roll: int = field(default=0)

    @staticmethod
    def get_dirs_8() -> Dict[str, "ViewOrientationExtended"]:
        directions = {
            "ftr": ViewOrientationExtended(
                46, 45, valid_sides=[ViewOrientation.FRONT, ViewOrientation.RIGHT, ViewOrientation.TOP]
            ),
            "btr": ViewOrientationExtended(
                45, 90 + 45, valid_sides=[ViewOrientation.TOP, ViewOrientation.RIGHT, ViewOrientation.BACK]
            ),
            "btl": ViewOrientationExtended(
                45, 180 + 45, valid_sides=[ViewOrientation.TOP, ViewOrientation.BACK, ViewOrientation.LEFT]
            ),
            "ftl": ViewOrientationExtended(
                45, 270 + 45, valid_sides=[ViewOrientation.TOP, ViewOrientation.LEFT, ViewOrientation.FRONT]
            ),
            "fbr": ViewOrientationExtended(
                -45, 45, valid_sides=[ViewOrientation.FRONT, ViewOrientation.RIGHT, ViewOrientation.BOTTOM]
            ),
            "bbr": ViewOrientationExtended(
                -45, 90 + 45, valid_sides=[ViewOrientation.RIGHT, ViewOrientation.BACK, ViewOrientation.BOTTOM]
            ),
            "bbl": ViewOrientationExtended(
                -45, 180 + 45, valid_sides=[ViewOrientation.BACK, ViewOrientation.LEFT, ViewOrientation.BOTTOM]
            ),
            "fbl": ViewOrientationExtended(
                -45, 270 + 45, valid_sides=[ViewOrientation.LEFT, ViewOrientation.FRONT, ViewOrientation.BOTTOM]
            ),
        }
        return directions

    @staticmethod
    def get_dirs_6() -> Dict[str, "ViewOrientationExtended"]:
        directions = {
            "fr": ViewOrientation.FRONT,
            "ba": ViewOrientation.BACK,
            "le": ViewOrientation.LEFT,
            "ri": ViewOrientation.RIGHT,
            "to": ViewOrientation.TOP,
            "bo": ViewOrientation.BOTTOM,
        }
        return directions

    @staticmethod
    def get_dir(dir: str) -> "ViewOrientationExtended":
        return ViewOrientationExtended.get_dirs_8()[dir]

    @staticmethod
    def from_orientation(orientation: ViewOrientation) -> "ViewOrientationExtended":
        if orientation == ViewOrientation.TOP:
            return ViewOrientationExtended(elevation=90, azimuth=1)
        elif orientation == ViewOrientation.BOTTOM:
            return ViewOrientationExtended(elevation=-90, azimuth=1, roll=180)
        elif orientation == ViewOrientation.LEFT:
            return ViewOrientationExtended(elevation=0, azimuth=-89)
        elif orientation == ViewOrientation.RIGHT:
            return ViewOrientationExtended(elevation=0, azimuth=91)
        elif orientation == ViewOrientation.FRONT:
            return ViewOrientationExtended(elevation=0, azimuth=1)
        elif orientation == ViewOrientation.BACK:
            return ViewOrientationExtended(elevation=0, azimuth=181)
        elif orientation == ViewOrientation.ISO:
            return ViewOrientationExtended(elevation=45, azimuth=46)
        else:
            raise NotImplementedError()

    @property
    def name(self):
        return "a%d-e%d" % (self.azimuth, self.elevation)

    def __hash__(self) -> int:
        return hash(self.name)
