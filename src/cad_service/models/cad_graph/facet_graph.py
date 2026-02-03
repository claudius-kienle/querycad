from dataclasses import dataclass

import numpy as np


@dataclass
class FacetGraph:
    vertices: np.ndarray
    edges: np.ndarray
