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

# Funzione per stimare i parametri iniziali
def estimate_initial_params(points):
    center_init = points.mean(axis=0)  # Centro della point cloud
    # Estima la distanza massima dal centro (raggio dell'oggetto)
    radius_est = np.max(np.linalg.norm(points - center_init, axis=1))  
    a = radius_est * 0.5  # Usa un valore ridotto per i semiassi iniziali
    b = radius_est * 0.5
    c = radius_est * 0.5
    theta = 0
    phi = 0
    return np.array([center_init[0], center_init[1], center_init[2], a, b, c, theta, phi])

# Funzione per normalizzare la point cloud
def normalize_point_cloud(points):
    center = points.mean(axis=0)  # Calcola il centro
    points -= center  # Centra i punti
    max_distance = np.max(np.linalg.norm(points, axis=1))  # Trova la distanza massima dal centro
    points /= max_distance  # Normalizza la point cloud
    return points

# Carica la point cloud
frame = "0000"
dir_path = "ball_pointclouds"
full_path = f"{dir_path}/pointcloud_{frame}.ply"
pcd_full = o3d.io.read_point_cloud(full_path)
points = np.asarray(pcd_full.points)

# Ridurre il numero di punti per il test
points = points[:1000]  # Limita ai primi 1000 punti

# Normalizza la point cloud
points = normalize_point_cloud(points)

# Fitting ellittico
params_init = estimate_initial_params(points)
options = {
    'max_nfev': 200,  # Numero massimo di iterazioni
    'xtol': 1e-6,     # Tolleranza di convergenza
    'ftol': 1e-6,     # Tolleranza per i residui
}

# Esegui l'ottimizzazione
res = least_squares(ellipsoid_model, params_init, args=(points,), **options)

# Visualizza i risultati
x0, y0, z0, a, b, c, theta, phi = res.x
print(f"Centro stimato: ({x0:.3f}, {y0:.3f}, {z0:.3f})")
print(f"Semiassi (a, b, c): ({a:.3f}, {b:.3f}, {c:.3f})")
print(f"Angoli di rotazione: (theta = {theta:.3f}, phi = {phi:.3f})")

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

# Creiamo le triangolazioni per la mesh
triangles = []
for i in range(x_rot.shape[0]-1):
    for j in range(y_rot.shape[1]-1):
        triangles.append([i * y_rot.shape[1] + j, (i+1) * y_rot.shape[1] + j, (i+1) * y_rot.shape[1] + (j+1)])
        triangles.append([i * y_rot.shape[1] + j, (i+1) * y_rot.shape[1] + (j+1), i * y_rot.shape[1] + (j+1)])

ellipsoid_mesh.triangles = o3d.utility.Vector3iVector(triangles)
ellipsoid_mesh.compute_vertex_normals()

# Visualizza la point cloud e l'ellissoide
o3d.visualization.draw_geometries([pcd_full, ellipsoid_mesh], window_name="Fitting Ellittico")
