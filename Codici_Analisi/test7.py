import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from trimesh.curvature import discrete_mean_curvature_measure

# === PARAMETRI ===
PLY_PATH = "ball_pointclouds/ball_frame_0005.ply"
POISSON_DEPTH = 9
CURVATURE_RADIUS = 0.01

# === STEP 1: Point cloud ===
pcd = o3d.io.read_point_cloud(PLY_PATH)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

# === STEP 2: Mesh Poisson (chiusa) ===
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)
mesh.compute_vertex_normals()

# === STEP 3: Conversione a Trimesh ===
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)

# === STEP 4: Curvatura media (H) ===
H = discrete_mean_curvature_measure(tri_mesh, tri_mesh.vertices, radius=CURVATURE_RADIUS)
H = H[np.isfinite(H)]

# === STEP 5: Risultati globali ===
mean_H = np.median(H)
mean_R = 1 / mean_H if mean_H > 0 else float('inf')

print(f"Mesh è chiusa: {mesh.is_edge_manifold() and mesh.is_watertight()}")
print(f"Media curvatura H: {mean_H:.5f} m⁻¹")
print(f"Media raggio stimato: {mean_R:.5f} m")

# === STEP 6: Visualizzazione colorata (vedo style) ===
import pyvista as pv

plotter = pv.Plotter()
pv_mesh = pv.PolyData(vertices, triangles)
pv_mesh["Mean curvature (H)"] = np.clip(H, -0.02, 0.02)  # taglio valori estremi

plotter.add_mesh(pv_mesh, scalars="Mean curvature (H)", cmap="viridis", show_edges=False)
plotter.add_scalar_bar(title="Curvatura media (H)")
plotter.show()
