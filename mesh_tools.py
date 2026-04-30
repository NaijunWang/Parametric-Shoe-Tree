

import trimesh
import trimesh.repair
import numpy as np
import sys
import os
import math
from shapely.geometry import Polygon


# 
# PART 1 - mesh cleaning

def clean_mesh(mesh):
    print(f"  Before: {len(mesh.vertices):,} verts | {len(mesh.faces):,} faces | watertight: {mesh.is_watertight}")

    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if mesh.is_watertight:
        print("  Fixed with trimesh.")
        print(f"  After:  {len(mesh.vertices):,} verts | {len(mesh.faces):,} faces | watertight: {mesh.is_watertight}")
        return mesh

    print("  Still not watertight — trying pymeshfix...")
    try:
        import pymeshfix
        mf  = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        mf.repair()
        pv  = mf.mesh
        fc  = pv.faces.reshape(-1, 4)[:, 1:]
        mesh = trimesh.Trimesh(vertices=np.array(pv.points), faces=fc, process=True)
    except ImportError:
        print("  pymeshfix not installed — run: pip install pymeshfix pyvista")

    print(f"  After:  {len(mesh.vertices):,} verts | {len(mesh.faces):,} faces | watertight: {mesh.is_watertight}")
    return mesh


## making clips

EMBED = 2.0   # how far the tab base sinks into the body 


def make_clip_tab(split_y, x_pos, z_pos, params):

    w = params["tab_width"]
    h = params["tab_height"]
    l = params["tab_length"]
    c = params["chamfer"]

    # Chamfered profile in XZ 
    profile = Polygon([
        (-w/2,       0),
        ( w/2,       0),
        ( w/2,       h),
        ( w/2 - c,   h + c),
        (-w/2 + c,   h + c),
        (-w/2,       h),
    ])

    tab = trimesh.creation.extrude_polygon(profile, height=l + EMBED)

    # Rotate
    r1 = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    tab.apply_transform(r1)

    r2 = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    tab.apply_transform(r2)

    tab.apply_translation([x_pos, split_y + l, z_pos])

    return tab


def make_clip_socket(split_y, x_pos, z_pos, params, tol=0.3):

    w = params["tab_width"]
    h = params["tab_height"] + params["chamfer"]
    l = params["tab_length"]

    profile = Polygon([
        (-w/2 - tol,  0),
        ( w/2 + tol,  0),
        ( w/2 + tol,  h + tol),
        (-w/2 - tol,  h + tol),
    ])

    socket = trimesh.creation.extrude_polygon(profile, height=l + tol)

    r1 = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    socket.apply_transform(r1)
    r2 = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    socket.apply_transform(r2)

    socket.apply_translation([x_pos, split_y + l + tol, z_pos])

    return socket


def add_snap_clips(half_a, half_b, mesh, split_y):
    """
    Place two clips inside the actual split face 
    """
    # find vertices that lie on the split plane
    tol = 0.01
    on_plane = np.abs(half_a.vertices[:, 1] - split_y) < tol
    face_verts = half_a.vertices[on_plane]

    if len(face_verts) < 3:
        print("  WARNING: could not find split face vertices, using bounds")
        x_min, _, z_min = half_a.bounds[0]
        x_max, _, z_max = half_a.bounds[1]
    else:
        x_min = face_verts[:, 0].min()
        x_max = face_verts[:, 0].max()
        z_min = face_verts[:, 2].min()
        z_max = face_verts[:, 2].max()

    cross_x  = x_max - x_min
    cross_z  = z_max - z_min
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2

    # insert clips well inside the split face 
    x_offset = cross_x * 0.25

    clip_params = {
        "tab_width"  : 8.0,
        "tab_height" : 3.0,
        "tab_length" : 10.0,
        "chamfer"    : 1.5,
    }

    print(f"  Split face: X=[{x_min:.1f}, {x_max:.1f}]  Z=[{z_min:.1f}, {z_max:.1f}]")
    print(f"  Clip positions: X = {x_center:.1f} ± {x_offset:.1f} mm  |  Z = {z_center:.1f} mm")
    print(f"  Tab base embedded {EMBED} mm into body for solid union")

    for x_sign in (+1, -1):
        x_pos = x_center + x_sign * x_offset

        tab    = make_clip_tab(split_y,    x_pos, z_center, clip_params)
        socket = make_clip_socket(split_y, x_pos, z_center, clip_params)

        print(f"  Unioning tab at X={x_pos:+.1f}...")
        half_a = trimesh.boolean.union([half_a, tab], engine="manifold")

        print(f"  Cutting socket at X={x_pos:+.1f}...")
        half_b = trimesh.boolean.difference([half_b, socket], engine="manifold")

    trimesh.repair.fix_normals(half_a)
    trimesh.repair.fix_normals(half_b)

    return half_a, half_b


def split_mesh_y(mesh, fraction=0.5):
    lo      = mesh.bounds[0][1]
    hi      = mesh.bounds[1][1]
    split_y = lo + fraction * (hi - lo)

    half_a = trimesh.intersections.slice_mesh_plane(mesh, [0, -1, 0], [0, split_y, 0], cap=True)
    half_b = trimesh.intersections.slice_mesh_plane(mesh, [0,  1, 0], [0, split_y, 0], cap=True)

    print(f"  Split at Y = {split_y:.1f} mm ({fraction*100:.0f}%)")
    return half_a, half_b, split_y



def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command  = sys.argv[1]
    filepath = sys.argv[2]

    if not os.path.exists(filepath):
        sys.exit(f"ERROR: file not found: {filepath}")

    prefix = os.path.splitext(filepath)[0]

    print(f"\nLoading {filepath}...")
    mesh = trimesh.load(filepath, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    print(f"  {len(mesh.vertices):,} vertices | {len(mesh.faces):,} faces")

    if command == "clean":
        print("\n--- Part 1: Cleaning mesh ---")
        mesh = clean_mesh(mesh)
        out  = f"{prefix}_clean.stl"
        mesh.export(out)
        print(f"\nSaved: {out}")

    elif command == "clips":
        print("\n--- Part 1: Cleaning mesh ---")
        mesh = clean_mesh(mesh)

        print("\n--- Splitting along Y axis ---")
        half_a, half_b, split_y = split_mesh_y(mesh, fraction=0.5)

        print("\n--- Part 2: Adding snap-fit clips ---")
        half_a, half_b = add_snap_clips(half_a, half_b, mesh, split_y)

        out_a = f"{prefix}_bottom.stl"
        out_b = f"{prefix}_top.stl"
        half_a.export(out_a)
        half_b.export(out_b)

        print(f"\nSaved: {out_a}  (bottom half — has tabs)")
        print(f"Saved: {out_b}  (top half    — has sockets)")


    else:
        sys.exit(f"Unknown command '{command}'. Use 'clean' or 'clips'.")


if __name__ == "__main__":
    main()