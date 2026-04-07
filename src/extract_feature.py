import numpy as np
import trimesh
import trimesh.transformations as tf
import os
import argparse


def align_mesh_via_pca(mesh):
    # Initial orientation fix for vertical scans
    if mesh.extents[2] > mesh.extents[1] and mesh.extents[2] > mesh.extents[0]:
        mesh.apply_transform(tf.rotation_matrix(np.radians(90), [1, 0, 0]))

    # Grounding and unit scaling (m to mm)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    l_raw = mesh.extents[1]
    scale = 1000.0 if l_raw < 1.0 else (10.0 if l_raw < 50.0 else 1.0)
    mesh.apply_scale(scale)

    # PCA to align longitudinal axis with Y
    mask = mesh.vertices[:, 2] < np.percentile(mesh.vertices[:, 2], 25)
    pts = mesh.vertices[mask][:, :2]
    pts_centered = pts - np.mean(pts, axis=0)
    cov = np.cov(pts_centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    main_axis = evecs[:, 1]
    angle = np.arctan2(main_axis[0], main_axis[1])
    mesh.apply_transform(tf.rotation_matrix(-angle, [0, 0, 1]))

    # Correct vertical flip using center of mass
    if mesh.center_mass[2] > (mesh.bounds[0][2] + mesh.extents[2] * 0.5):
        mesh.apply_transform(tf.rotation_matrix(np.radians(180), [0, 1, 0]))

    # Correct longitudinal flip using thickness heuristic
    def get_max_z_span(m, y_range):
        v_subset = m.vertices[
            (m.vertices[:, 1] > y_range[0]) & (m.vertices[:, 1] < y_range[1])
        ]
        return np.ptp(v_subset[:, 2]) if len(v_subset) > 0 else 0

    b = mesh.bounds
    y_len = b[1][1] - b[0][1]
    front_thickness = get_max_z_span(mesh, [b[1][1] - y_len * 0.15, b[1][1]])
    back_thickness = get_max_z_span(mesh, [b[0][1], b[0][1] + y_len * 0.15])

    if front_thickness > back_thickness:
        mesh.apply_transform(tf.rotation_matrix(np.radians(180), [0, 0, 1]))

    # Final normalization
    final_b = mesh.bounds
    center_x = (final_b[0][0] + final_b[1][0]) / 2
    mesh.apply_translation([-center_x, -final_b[0][1], -final_b[0][2]])

    return mesh


def get_basic_dims(mesh):
    return {
        "length": round(float(mesh.extents[1]), 2),
        "width": round(float(mesh.extents[0]), 2),
        "height": round(float(mesh.extents[2]), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Foot Feature Extractor - Exhibition Version"
    )
    parser.add_argument("model_path", help="Path to mesh file")
    parser.add_argument("--no-show", action="store_true", help="Disable viewer")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: {args.model_path} not found")
        return

    mesh = trimesh.load(args.model_path)
    mesh = align_mesh_via_pca(mesh)
    dims = get_basic_dims(mesh)

    # Define Slice Positions
    ball_y = dims["length"] * 0.72  # Toe-Instep connection
    instep_y = dims["length"] * 0.50  # Instep peak

    # Calculate Sections
    sec_ball = mesh.section(plane_origin=[0, ball_y, 0], plane_normal=[0, 1, 0])
    sec_instep = mesh.section(plane_origin=[0, instep_y, 0], plane_normal=[0, 1, 0])

    print("-" * 60)
    print(f"File Name : {os.path.basename(args.model_path)}")
    print("-" * 60)
    print("[Basic Dimensions]")
    print(f"Total Length (Y): {dims['length']} mm")
    print(f"Max Width    (X): {dims['width']} mm")
    print(f"Max Height   (Z): {dims['height']} mm")
    print("\n[Section Features]")
    if sec_ball:
        print(
            f"1. Toe-Instep Perimeter (Red):    {sec_ball.length:.2f} mm (at Y={ball_y:.1f})"
        )
    if sec_instep:
        print(
            f"2. Instep Peak Perimeter (Yellow): {sec_instep.length:.2f} mm (at Y={instep_y:.1f})"
        )
    print("-" * 60)

    if not args.no_show:
        vis_mesh = mesh.copy()
        vis_mesh.visual = trimesh.visual.ColorVisuals(mesh=vis_mesh)
        vis_mesh.visual.face_colors = [200, 200, 200, 80]  # Semi-transparent

        axis = trimesh.creation.axis(origin_size=3, axis_radius=1.5, axis_length=300)
        scene_elements = [vis_mesh, axis]

        # Visualization: Ball/Toe Slice (RED)
        if sec_ball and len(sec_ball.entities) > 0:
            sec_ball.colors = np.tile([255, 0, 0, 255], (len(sec_ball.entities), 1))
            p_ball = trimesh.creation.box(
                extents=[dims["width"] * 1.5, 0.2, dims["height"] * 1.2]
            )
            p_ball.apply_translation([0, ball_y, dims["height"] / 2])
            p_ball.visual.face_colors = [255, 0, 0, 40]
            scene_elements.extend([sec_ball, p_ball])

        # Visualization: Instep Peak Slice (YELLOW)
        if sec_instep and len(sec_instep.entities) > 0:
            sec_instep.colors = np.tile(
                [255, 255, 0, 255], (len(sec_instep.entities), 1)
            )
            p_instep = trimesh.creation.box(
                extents=[dims["width"] * 1.5, 0.2, dims["height"] * 1.2]
            )
            p_instep.apply_translation([0, instep_y, dims["height"] / 2])
            p_instep.visual.face_colors = [255, 255, 0, 40]
            scene_elements.extend([sec_instep, p_instep])

        trimesh.Scene(scene_elements).show()


if __name__ == "__main__":
    main()
