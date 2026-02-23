import open3d as o3d
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from vedo import Plotter, Mesh

# === STEP 1: Carica e filtra la point cloud ===
frame = "0005"
pcd = o3d.io.read_point_cloud(f"ball_pointclouds/ball_frame_{frame}.ply")
points = np.asarray(pcd.points)

# Radiale + DBSCAN
center = points.mean(axis=0)
points = points[np.linalg.norm(points - center, axis=1) < 0.08]

labels = DBSCAN(eps=0.0085, min_samples=20).fit(points).labels_
mask = labels == np.argmax(np.bincount(labels[labels != -1]))
points = points[mask]

pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals()

# === STEP 2: Mesh via ball pivoting ===
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radii = [3 * avg_dist, 5 * avg_dist]
""" mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
 """
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
mesh = mesh.remove_unreferenced_vertices()
mesh.orient_triangles()
print("Mesh è chiusa:", mesh.is_edge_manifold() and mesh.is_watertight())

# === STEP 3: Converte in trimesh ===
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# === STEP 4: Curvatura media + gaussiana ===
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure

radius = 0.01
gauss = discrete_gaussian_curvature_measure(tmesh, tmesh.vertices, radius=radius)
mean = discrete_mean_curvature_measure(tmesh, tmesh.vertices, radius=radius)

# === STEP 5: Visualizza con vedo ===
m = Mesh([vertices, faces])
m.pointdata["H (mean)"] = mean
m.pointdata["K (gauss)"] = gauss
m.cmap("viridis", "H (mean)").add_scalarbar(title="Curvatura media (H)")

vp = Plotter(title="Curvatura media - Mesh pallina")
vp.show(m, axes=1, interactive=True)

print(f"Media curvatura H: {np.median(mean):.5f} m⁻¹")
print(f"Media raggio stimato: {1/np.median(mean):.5f} m")
