"""推論引擎層 — model-agnostic Python API（推論的唯一入口）。"""
from lbmnca.nca.inference.geometry_types import GeometryInputs
from lbmnca.nca.inference.engine import InferenceEngine

__all__ = ["GeometryInputs", "InferenceEngine"]
