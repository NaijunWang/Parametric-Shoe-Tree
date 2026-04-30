

import trimesh
import numpy as np
import argparse
import sys
import os


def load_mesh(path):
    print(f"Loading {path}...")
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    print(f"  {len(mesh.vertices):,} vertices  |  {len(mesh.faces):,} faces")
    return mesh


def split_mesh(mesh, fraction=0.5, axis="x"):

    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    lo        = mesh.bounds[0][axis_index]
    hi        = mesh.bounds[1][axis_index]
    split_val = lo + fraction * (hi - lo)

    plane_origin = [0.0, 0.0, 0.0]
    plane_origin[axis_index] = split_val

    normal_pos = [0.0, 0.0, 0.0]
    normal_neg = [0.0, 0.0, 0.0]
    normal_pos[axis_index] =  1.0  
    normal_neg[axis_index] = -1.0  

    print(f"Splitting along {axis.upper()} at {split_val:.1f} mm ({fraction*100:.0f}% of range)...")

    half_a = trimesh.intersections.slice_mesh_plane(
        mesh, normal_neg, plane_origin, cap=True
    )
    half_b = trimesh.intersections.slice_mesh_plane(
        mesh, normal_pos, plane_origin, cap=True
    )

    return half_a, half_b


def main():
    ap = argparse.ArgumentParser(description="Split a mesh in half for print bed.")
    ap.add_argument("file",           help="Input mesh (OBJ, STL, PLY)")
    ap.add_argument("--split", type=float, default=0.5,
                    help="Split fraction 0–1  (default: 0.5 = midpoint)")
    ap.add_argument("--axis",  type=str,   default="x",
                    choices=["x", "y", "z"],
                    help="Axis to split along  (default: x = length)")
    ap.add_argument("--out",   type=str,   default=None,
                    help="Output prefix  (default: input filename)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"ERROR: file not found: {args.file}")

    prefix = args.out or os.path.splitext(args.file)[0]

    mesh = load_mesh(args.file)
    a, b = split_mesh(mesh, fraction=args.split, axis=args.axis)

    out_a = f"{prefix}_half_a.stl"
    out_b = f"{prefix}_half_b.stl"
    a.export(out_a)
    b.export(out_b)

    extents  = mesh.bounding_box.extents
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]

    print(f"\nSaved: {out_a}  ({len(a.faces):,} faces, watertight: {a.is_watertight})")
    print(f"Saved: {out_b}  ({len(b.faces):,} faces, watertight: {b.is_watertight})")
    print(f"\nOriginal bounding box: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"Each half: ~{extents[axis_idx] * args.split / 25.4:.2f}\" along the split axis")


if __name__ == "__main__":
    main()