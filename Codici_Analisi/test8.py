import open3d as o3d
import numpy as np
import glob
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# === Lettura point cloud dal folder
files = sorted(glob.glob("ball_pointclouds/ball_frame_0005.ply"))
all_points = []

for file in files:
    pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(pcd.points)
    if len(points) > 0:
        all_points.append(points)

if len(all_points) == 0:
    print("[ERRORE] Nessuna point cloud trovata!")
    exit()

points = np.vstack(all_points)
print(f"[INFO] Point cloud totale: {points.shape[0]} punti")

# === Pulizia: filtro radiale + DBSCAN ===
center = points.mean(axis=0)
mask_radial = np.linalg.norm(points - center, axis=1) < 0.08
points = points[mask_radial]
print(f"[INFO] Dopo filtro radiale: {points.shape[0]} punti")

labels = DBSCAN(eps=0.0085, min_samples=20).fit(points).labels_
mask_dbscan = labels == np.argmax(np.bincount(labels[labels != -1]))
points = points[mask_dbscan]
print(f"[INFO] Dopo DBSCAN: {points.shape[0]} punti")

# === Salva la point cloud filtrata
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("merged_ball_filtered.ply", pcd_filtered)
print("[INFO] Salvata point cloud filtrata: merged_ball_filtered.ply")

# === Calcolo curvature
tree = KDTree(points)
curvatures = []
k_neighbors = 50

for i, p in enumerate(points):
    dists, idx = tree.query([p], k=k_neighbors+1)
    neighbors = points[idx[0,1:]]
    centered = neighbors - neighbors.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    curvature = eigvals[0] / eigvals.sum()
    curvatures.append(curvature)

curvatures = np.array(curvatures)
curv_norm = (curvatures - curvatures.min()) / (curvatures.ptp() + 1e-8)
colors = plt.get_cmap("jet")(curv_norm)[:,:3]
pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

# === Visualizzazione con mappa colori curvature
o3d.visualization.draw_geometries([pcd_filtered], window_name="Curvature Map")

# === Statistiche globali
mean_curv = np.median(curvatures)
mean_radius = 1 / mean_curv if mean_curv > 1e-8 else np.inf
print("\n=== ANALISI CURVATURA ===")
print(f"Curvatura media (H)  : {mean_curv:.5f}")
print(f"Raggio stimato (1/H) : {mean_radius:.5f} m")
