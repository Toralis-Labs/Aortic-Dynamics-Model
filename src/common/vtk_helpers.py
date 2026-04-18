from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import vtk
from vtk.util import numpy_support


def read_vtp(path: str | Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    return output


def write_vtp(polydata: vtk.vtkPolyData, path: str | Path, binary: bool = True) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(polydata)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {out_path}")


def array_names(attrs: Any) -> list[str]:
    names: list[str] = []
    for idx in range(attrs.GetNumberOfArrays()):
        arr = attrs.GetArray(idx)
        if arr is not None:
            names.append(str(arr.GetName()))
    return names


def get_cell_array(polydata: vtk.vtkPolyData, name: str) -> Optional[np.ndarray]:
    arr = polydata.GetCellData().GetArray(name)
    if arr is None:
        return None
    return numpy_support.vtk_to_numpy(arr)


def get_point_array(polydata: vtk.vtkPolyData, name: str) -> Optional[np.ndarray]:
    arr = polydata.GetPointData().GetArray(name)
    if arr is None:
        return None
    return numpy_support.vtk_to_numpy(arr)


def points_to_numpy(polydata: vtk.vtkPolyData) -> np.ndarray:
    pts = polydata.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return numpy_support.vtk_to_numpy(pts.GetData()).astype(float, copy=True)


def cell_centers(polydata: vtk.vtkPolyData) -> np.ndarray:
    centers = vtk.vtkCellCenters()
    centers.SetInputData(polydata)
    centers.Update()
    return points_to_numpy(centers.GetOutput())


def clone_geometry_only(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    out = vtk.vtkPolyData()
    out.DeepCopy(polydata)
    out.GetPointData().Initialize()
    out.GetCellData().Initialize()
    out.GetFieldData().Initialize()
    return out


def add_int_cell_array(polydata: vtk.vtkPolyData, name: str, values: Iterable[int]) -> None:
    arr = vtk.vtkIntArray()
    arr.SetName(name)
    vals = list(values)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(vals))
    for idx, value in enumerate(vals):
        arr.SetValue(idx, int(value))
    polydata.GetCellData().AddArray(arr)


def add_uchar3_cell_array(polydata: vtk.vtkPolyData, name: str, values: Iterable[tuple[int, int, int]]) -> None:
    arr = vtk.vtkUnsignedCharArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(3)
    vals = list(values)
    arr.SetNumberOfTuples(len(vals))
    for idx, rgb in enumerate(vals):
        arr.SetTuple3(idx, int(rgb[0]), int(rgb[1]), int(rgb[2]))
    polydata.GetCellData().AddArray(arr)
    polydata.GetCellData().SetActiveScalars(name)


def add_int_point_array(polydata: vtk.vtkPolyData, name: str, values: Iterable[int]) -> None:
    arr = vtk.vtkIntArray()
    arr.SetName(name)
    vals = list(values)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(vals))
    for idx, value in enumerate(vals):
        arr.SetValue(idx, int(value))
    polydata.GetPointData().AddArray(arr)


def build_polyline_polydata(points: np.ndarray) -> vtk.vtkPolyData:
    pts_np = np.asarray(points, dtype=float)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(int(pts_np.shape[0]))
    for idx, p in enumerate(pts_np):
        vtk_points.SetPoint(idx, float(p[0]), float(p[1]), float(p[2]))
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(int(pts_np.shape[0]))
    for idx in range(int(pts_np.shape[0])):
        polyline.GetPointIds().SetId(idx, idx)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)
    out = vtk.vtkPolyData()
    out.SetPoints(vtk_points)
    out.SetLines(cells)
    return out


def build_segment_point_locator(segment_points: list[tuple[int, np.ndarray]]) -> tuple[vtk.vtkStaticPointLocator, vtk.vtkPolyData]:
    vtk_points = vtk.vtkPoints()
    segment_ids: list[int] = []
    for segment_id, pts in segment_points:
        for p in np.asarray(pts, dtype=float):
            vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            segment_ids.append(int(segment_id))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_points)
    add_int_point_array(pd, "SegmentId", segment_ids)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(pd)
    locator.BuildLocator()
    return locator, pd


def segment_color(segment_id: int) -> tuple[int, int, int]:
    if int(segment_id) <= 0:
        return (128, 128, 128)
    hue = ((int(segment_id) * 0.61803398875) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def triangle_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    if pts.shape[0] == 3:
        return 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
    anchor = pts[0]
    area = 0.0
    for idx in range(1, pts.shape[0] - 1):
        area += 0.5 * float(np.linalg.norm(np.cross(pts[idx] - anchor, pts[idx + 1] - anchor)))
    return area if math.isfinite(area) else 0.0


def cell_points(polydata: vtk.vtkPolyData, cell_id: int) -> np.ndarray:
    cell = polydata.GetCell(int(cell_id))
    ids = cell.GetPointIds()
    pts = []
    for idx in range(ids.GetNumberOfIds()):
        pts.append(polydata.GetPoint(ids.GetId(idx)))
    return np.asarray(pts, dtype=float)

