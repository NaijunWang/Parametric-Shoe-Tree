import numpy as np
import trimesh
import trimesh.transformations as tf
import os
import argparse

def get_area_from_path3d(path3d):
    if path3d is None:
        return 0
    try:
        path2d, _ = path3d.to_2D()
        return path2d.area
    except:
        return 0

def align_mesh(mesh):
    # Normalize scale to millimeters based on bounding box size
    max_ext = max(mesh.extents)
    scale = 1000.0 if max_ext < 1.0 else (10.0 if max_ext < 50.0 else 1.0)
    mesh.apply_scale(scale)

    # get the basic box alignment
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)

    # the longest side is the y axis
    ext = mesh.extents
    long_axis = np.argmax(ext)
    if long_axis == 0: 
        mesh.apply_transform(tf.rotation_matrix(np.radians(90), [0, 0, 1]))
    elif long_axis == 2: 
        mesh.apply_transform(tf.rotation_matrix(np.radians(90), [1, 0, 0]))

    def get_section_area_at_axis(axis_idx):
        try:
            origin = [0, 0, 0]
            origin[axis_idx] = mesh.centroid[axis_idx]
            normal = [0, 0, 0]
            normal[axis_idx] = 1
            sec = mesh.section(plane_origin=origin, plane_normal=normal)
            return get_area_from_path3d(sec)
        except:
            return 0

    # compare projected surface areas
    area_x = get_section_area_at_axis(0) # side view area
    area_z = get_section_area_at_axis(2) # footprint area

    # Rotate 90 deg around Y 
    if area_x > area_z:
        mesh.apply_transform(tf.rotation_matrix(np.radians(90), [0, 1, 0]))

    b = mesh.bounds
    z_min_sec = mesh.section(plane_origin=[0, 0, b[0][2] + 5], plane_normal=[0, 0, 1])
    z_max_sec = mesh.section(plane_origin=[0, 0, b[1][2] - 5], plane_normal=[0, 0, 1])
    area_bottom = get_area_from_path3d(z_min_sec)
    area_top = get_area_from_path3d(z_max_sec)
    
    if area_top > area_bottom:
        mesh.apply_transform(tf.rotation_matrix(np.radians(180), [1, 0, 0]))

    # Orient the toes toward +y axis
    curr_b = mesh.bounds
    y_len = curr_b[1][1] - curr_b[0][1]
    f_pts = mesh.vertices[mesh.vertices[:, 1] > (curr_b[1][1] - y_len * 0.15)]
    b_pts = mesh.vertices[mesh.vertices[:, 1] < (curr_b[0][1] + y_len * 0.15)]
    f_thick = np.ptp(f_pts[:, 2]) if len(f_pts) > 0 else 0
    b_thick = np.ptp(b_pts[:, 2]) if len(b_pts) > 0 else 0

    if f_thick > b_thick :
        mesh.apply_transform(tf.rotation_matrix(np.radians(180), [0, 0, 1]))

    v = mesh.vertices
    min_v, max_v = v.min(axis=0), v.max(axis=0)
    v[:, 0] -= (min_v[0] + max_v[0]) / 2
    v[:, 1] -= min_v[1]
    v[:, 2] -= min_v[2]
    mesh.vertices = v
    mesh._cache.clear() 
    
    return mesh

def get_basic_dims(mesh):
    return {
        "length": round(float(mesh.extents[1]), 2),
        "width": round(float(mesh.extents[0]), 2),
        "height": round(float(mesh.extents[2]), 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Foot Feature Extractor")
    parser.add_argument("model_path", help="Path to mesh file")
    parser.add_argument("--no-show", action="store_true", help="Disable viewer")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: {args.model_path} not found")
        return

    mesh = trimesh.load(args.model_path)
    mesh = align_mesh(mesh)
    
    dims = get_basic_dims(mesh)
    ball_y = dims["length"] * 0.72
    instep_y = dims["length"] * 0.50

    sec_ball = mesh.section(plane_origin=[0, ball_y, 0], plane_normal=[0, 1, 0])
    sec_instep = mesh.section(plane_origin=[0, instep_y, 0], plane_normal=[0, 1, 0])

    print("-" * 60)
    print(f"File Name : {os.path.basename(args.model_path)}")
    print("-" * 60)
    print(f"Total Length (Y): {dims['length']} mm")
    print(f"Max Width    (X): {dims['width']} mm")
    print(f"Max Height   (Z): {dims['height']} mm")
    print(f"Toe-Instep Perimeter: {sec_ball.length:.2f} mm")
    print(f"Instep Peak Perimeter: {sec_instep.length:.2f} mm")
    print("-" * 60)

    if not args.no_show:
        vis_mesh = mesh.copy()
        vis_mesh.visual.face_colors = [200, 200, 200, 80]
        axis = trimesh.creation.axis(origin_size=5, axis_radius=2, axis_length=300)
        scene_elements = [vis_mesh, axis]
        
        for y, color in zip([ball_y, instep_y], [[255,0,0,100], [255,255,0,100]]):
            p = trimesh.creation.box(extents=[dims['width']*1.5, 0.5, dims['height']*1.2])
            p.apply_translation([0, y, dims['height']/2])
            p.visual.face_colors = color
            scene_elements.append(p)

        trimesh.Scene(scene_elements).show()

if __name__ == "__main__":
    main()
