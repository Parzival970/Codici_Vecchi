import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from open3d.geometry import PointCloud, TriangleMesh

# Funzione per il fitting sferico (già visto)
def fit_sphere(points):
    A = np.hstack((2 * points, np.ones((len(points), 1))))
    f = np.sum(points**2, axis=1).reshape(-1, 1)
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3].flatten()
    radius = np.sqrt(C[3] + np.sum(center**2))
    return center, radius

# Funzione per il fitting cilindrico
def fit_cylinder(points):
    # Approssimazione cilindrica (per semplificare, useremo un fitting su piani verticali)
    # Useremo RANSAC per il fitting cilindrico in un piano
    pass  # Implementare il fitting cilindrico qui

# Funzione per il fitting piano
def fit_plane(points):
    plane_model, inliers = points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    return plane_model  # [a, b, c, d] per l'equazione del piano ax + by + cz + d = 0

# === Carica la point cloud ===
frame = "0000"
dir_path = "ball_pointclouds"
full_path = f"{dir_path}/pointcloud_{frame}.ply"
pcd_full = o3d.io.read_point_cloud(full_path)
points = np.asarray(pcd_full.points)

# === Segmentazione con DBSCAN ===
eps = 0.0085  # Parametro per la segmentazione
min_samples = 20
labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

# Filtra oggetti validi
unique_labels = set(labels)
unique_labels = unique_labels - {-1}  # Rimuove il rumore (-1)
pcd_objects = []

for label in unique_labels:
    cluster_points = points[labels == label]
    pcd_objects.append(cluster_points)  # Ogni oggetto è un cluster separato

# === Approssimazione per ogni oggetto ===
for i, obj_points in enumerate(pcd_objects):
    print(f"\nFitting dell'oggetto {i + 1}...")

    # Se l'oggetto è approssimabile come una sfera
    center, radius = fit_sphere(obj_points)
    print(f"Centro sfera: {center}, Raggio: {radius}")

    # Fitting di altre primitive (cilindro, piano, etc.)
    # Aggiungi il codice per cilindro o piano qui, come necessario

    # Visualizza i risultati
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(obj_points)
    o3d.visualization.draw_geometries([pcd_object], window_name=f"Oggetto {i + 1}")

    # Visualizza la sfera stimata
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.translate(center)
    o3d.visualization.draw_geometries([pcd_object, sphere_mesh], window_name=f"Sfera stimata - Oggetto {i + 1}")

