import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import open3d as o3d
import matplotlib.pyplot as plt

# === 1) Caricamento point cloud ===
pcd = o3d.io.read_point_cloud("ball_pointclouds/ball_frame_0005.ply")
points = np.asarray(pcd.points)

# === 2) Filtraggio: filtro radiale + DBSCAN ===
# Filtro radiale: elimina punti lontani dal centro stimato
center = points.mean(axis=0)
mask_radial = np.linalg.norm(points - center, axis=1) < 0.08
points = points[mask_radial]
print(f"[INFO] Dopo filtro radiale: {points.shape[0]} punti")

# DBSCAN: rimuove rumore sparso, tiene solo cluster principale
labels = DBSCAN(eps=0.0085, min_samples=20).fit(points).labels_
mask_dbscan = labels == np.argmax(np.bincount(labels[labels != -1]))
points = points[mask_dbscan]
print(f"[INFO] Dopo DBSCAN: {points.shape[0]} punti")

# Aggiorna point cloud filtrata
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# === 3) Calcolo curvature principali punto per punto ===
tree = KDTree(points)
k_neighbors = 50
curvatures = []

for i, p in enumerate(points):
    dists, idx = tree.query(p, k=k_neighbors+1)
    neighbor_idx = idx[1:]
    neighbors = points[neighbor_idx]
    
    neighbors_centered = neighbors - p
    
    # Normale stimata via PCA
    cov = np.cov(neighbors_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal /= np.linalg.norm(normal)
    
    # Sistema locale u,v ortogonale a normal
    ref_axis = np.array([1,0,0]) if abs(normal[2]) > 0.9 else np.array([0,0,1])
    u_axis = np.cross(normal, ref_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    
    # Coordinate dei vicini nel sistema (u,v,n)
    U = neighbors_centered.dot(u_axis)
    V = neighbors_centered.dot(v_axis)
    W = neighbors_centered.dot(normal)
    
    A_mat = np.column_stack([U**2, U*V, V**2, U, V])
    b_vec = W
    coeff, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    A,B,C,D,E = coeff
    
    H = np.array([[2*A, B],
                  [B, 2*C]])
    k1, k2 = np.linalg.eigvals(H)
    k1, k2 = sorted([k1, k2], key=lambda x: abs(x), reverse=True)
    
    R1 = np.inf if abs(k1) < 1e-8 else 1.0/k1
    R2 = np.inf if abs(k2) < 1e-8 else 1.0/k2
    
    curvatures.append((k1, k2, R1, R2))

# === 4) Mappa di calore con curvatura gaussiana ===
k1_vals = np.array([k[0] for k in curvatures])
k2_vals = np.array([k[1] for k in curvatures])
K_vals = k1_vals * k2_vals  # curvatura gaussiana

# Normalizza K a [0,1] evitando outlier estremi
K_min, K_max = np.percentile(K_vals, [1, 99])
K_norm = np.clip((K_vals - K_min) / (K_max - K_min), 0, 1)

# Mappa i valori normalizzati a colori (colormap viridis)
colormap = plt.cm.get_cmap('viridis')
colors = colormap(K_norm)[:, :3]

# Assegna colori alla point cloud e visualizza
pcd.colors = o3d.utility.Vector3dVector(colors)

# Mostra la colorbar della mappa di calore
fig, ax = plt.subplots(figsize=(6, 1))
norm = plt.Normalize(K_min, K_max)
cb1 = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=colormap),
    cax=ax, orientation='horizontal'
)
cb1.set_label('Curvatura Gaussiana')
plt.title('Colorbar: valori di curvatura gaussiana')
plt.tight_layout()
plt.show()

o3d.visualization.draw_geometries([pcd])
