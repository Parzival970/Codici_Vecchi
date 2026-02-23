import open3d as o3d
import numpy as np
import sys
from sklearn.cluster import DBSCAN

# === Parametri ===
frame = sys.argv[1] if len(sys.argv) > 1 else "0000"
dir_path = "ball_pointclouds"
full_path = f"{dir_path}/pointcloud_{frame}.ply"
ball_path = f"{dir_path}/ball_frame_{frame}.ply"
print(ball_path)

radius_max = 0.08       # m dal centro
z_offset = 0.015        # m sotto il centro (per tagliare tavolo)
eps = 0.0085         # DBSCAN
min_samples = 20
outlier_threshold = 0.004  # 3 mm

# === Carica point cloud ===
pcd_full = o3d.io.read_point_cloud(full_path)
pcd_ball = o3d.io.read_point_cloud(ball_path)
o3d.visualization.draw_geometries([pcd_ball], window_name=f"Frame {frame} - Point Cloud")

points = np.asarray(pcd_ball.points)
if len(points) < 10:
    print("[WARN] Point cloud troppo piccola.")
    exit()

# === Filtro radiale dal centro ===
center_init = points.mean(axis=0)
dists = np.linalg.norm(points - center_init, axis=1)
points = points[dists < radius_max]

# === DBSCAN clustering ===
if len(points) > 10:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    if np.any(labels != -1):
        majority = np.argmax(np.bincount(labels[labels != -1]))
        points = points[labels == majority]

""" # === Filtro Z per togliere il tavolo ===
z_median = np.median(points[:, 2])
points = points[points[:, 2] > z_median - z_offset] """

# === Fitting sferico ===
A = np.hstack((2 * points, np.ones((len(points), 1))))
f = np.sum(points**2, axis=1).reshape(-1, 1)
C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
center = C[:3].flatten()
radius = float(np.sqrt(C[3] + np.sum(center**2)))
dists_to_center = np.linalg.norm(points - center, axis=1)
rmse = np.sqrt(np.mean((dists_to_center - radius) ** 2))

# === Rimozione outlier dopo fitting ===
errors = np.abs(dists_to_center - radius)
mask = errors < outlier_threshold
points_clean = points[mask]

print("\n=== RISULTATI FINALI ===")
print(f"Centro stimato       : {center}")
print(f"Raggio stimato       : {radius:.5f} m")
print(f"Curvatura media (H)  : {1/radius:.2f} m⁻¹")
print(f"Curvatura gaussiana  : {1/(radius**2):.2f} m⁻²")
print(f"Errore medio (RMSE)  : {rmse*1000:.3f} mm")

# === Visualizza point cloud filtrata + sfera stimata ===
pcd_scene = pcd_full
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points_clean)
pcd_clean.paint_uniform_color([1, 0, 0])  # rosso
o3d.visualization.draw_geometries([pcd_clean], window_name=f"Frame {frame} - Pallina filtrata + Sfera stimata")
# === Crea mesh della sfera stimata ===
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
sphere.translate(center)
sphere.compute_vertex_normals()
sphere.paint_uniform_color([0.2, 0.7, 1.0])  # azzurra trasparente
sphere = sphere.subdivide_midpoint(1)

#o3d.visualization.draw_geometries([pcd_clean,sphere], window_name=f"Pallina filtrata - Frame {frame}")
# === Visualizzazione ===
o3d.visualization.draw_geometries([pcd_scene, pcd_clean, sphere],window_name=f"Frame {frame} – Pallina filtrata + Sfera stimata")

o3d.visualization.draw_geometries([pcd_scene, pcd_clean],window_name=f"Frame {frame} – Pallina filtrata + Sfera stimata")
