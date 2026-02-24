"""Generate shirt.npz, pants.npz, skirt.npz from gt meshes (y_center, diag_max). Run from ISP repo root."""
import os
import sys
import numpy as np
import trimesh

def main():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
    if not os.path.isdir(os.path.join(root, "extra-data")):
        root = os.getcwd()
    data_path = os.path.join(root, "extra-data")
    meshes = [
        ("shirt.npz", "meshes/tee-gt.obj"),
        ("pants.npz", "meshes/pants-gt.obj"),
        ("skirt.npz", "meshes/skirt-gt.obj"),
    ]
    for npz_name, mesh_rel in meshes:
        mesh_path = os.path.join(data_path, mesh_rel)
        if not os.path.isfile(mesh_path):
            print("Skip (not found):", mesh_path)
            continue
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        y_center = float(np.mean(verts[:, 1]))
        bounds = mesh.bounds
        diag_max = float(np.linalg.norm(bounds[1] - bounds[0]))
        if diag_max <= 0:
            diag_max = 1.0
        out_path = os.path.join(data_path, npz_name)
        np.savez(out_path, y_center=np.array(y_center), diag_max=np.array(diag_max))
        print("Wrote", out_path, "y_center=", y_center, "diag_max=", diag_max)

if __name__ == "__main__":
    main()
