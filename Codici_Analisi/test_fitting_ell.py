import open3d as o3d
import numpy as np
from scipy.optimize import least_squares

# Funzione per l'ellissoide 3D
def ellipsoid_model(params, points):
    x0, y0, z0, a, b, c, theta, phi = params
    x, y, z = points.T
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Ruotiamo i punti attorno agli assi
    x_rot = cos_phi * cos_theta * (x - x0) + sin_phi * (y - y0)
    y_rot = -sin_phi * cos_theta * (x - x0) + cos_phi * (y - y0)
    z_rot = sin_theta * (x - x0) + cos_theta * (z - z0)

    # Modello ellissoidale
    return ((x_rot / a) ** 2 + (y_rot / b) ** 2 + (z_rot / c) ** 2 - 1)

# Funzione per il fitting
def fit_ellipsoid(points):
    # Parametri iniziali
    params_init = np.array([0, 0, 0, 1, 1, 1, 0, 0])  # [x0, y0, z0, a, b, c, theta, phi]
    
    # Uso del least squares per ottimizzare i parametri
    res = least_squares(ellipsoid_model, params_init, args=(points,))
    return res.x

# === Carica point cloud ===
frame = "0000"  # Usa un nome di frame appropriato
dir_path = "ball_pointclouds"
full_path = f"{dir_path}/pointcloud_{frame}.ply"
pcd_full = o3d.io.read_point_cloud(full_path)

# Estrai i punti dalla point cloud
points = np.asarray(pcd_full.points)

# === Rimozione outlier ===
# Usa il DBSCAN come nel tuo codice per rimuovere i punti errati (opzionale)
from sklearn.cluster import DBSCAN
eps = 0.0085
min_samples = 20
labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
if np.any(labels != -1):
    majority = np.argmax(np.bincount(labels[labels != -1]))
    points = points[labels == majority]

# === Fitting ellittico ===
params = fit_ellipsoid(points)
x0, y0, z0, a, b, c, theta, phi = params

print("\n=== RISULTATI FITTING ELLITTICO ===")
print(f"Centro stimato       : ({x0:.3f}, {y0:.3f}, {z0:.3f})")
print(f"Semiassi (a, b, c)   : ({a:.3f}, {b:.3f}, {c:.3f})")
print(f"Angoli di rotazione  : (theta = {theta:.3f}, phi = {phi:.3f})")

# === Visualizzazione ===
# Crea la mesh dell'ellissoide
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
x = a * np.cos(u) * np.sin(v)
y = b * np.sin(u) * np.sin(v)
z = c * np.cos(v)

# Ruotiamo e trasliamo l'ellissoide
x_rot = x * np.cos(phi) * np.cos(theta) + y * np.sin(phi)
y_rot = -x * np.sin(phi) * np.cos(theta) + y * np.cos(phi)
z_rot = x * np.sin(theta) + z * np.cos(theta)

# Traslazione dell'ellissoide
x_rot += x0
y_rot += y0
z_rot += z0

# Crea la mesh dell'ellissoide
ellipsoid_mesh = o3d.geometry.TriangleMesh()
ellipsoid_mesh.vertices = o3d.utility.Vector3dVector(np.vstack((x_rot.flatten(), y_rot.flatten(), z_rot.flatten())).T)
ellipsoid_mesh.triangles = o3d.utility.Vector3iVector(np.array([[i, i + 1, i + 2] for i in range(len(x_rot.flatten()) - 2)]))
ellipsoid_mesh.compute_vertex_normals()

# Visualizza
o3d.visualization.draw_geometries([pcd_full, ellipsoid_mesh], window_name="Fitting Ellittico")
