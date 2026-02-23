import open3d as o3d
import numpy as np
from scipy.optimize import least_squares

# Funzione per il fitting sferico (già visto)
def fit_sphere(points):
    A = np.hstack((2 * points, np.ones((len(points), 1))))
    f = np.sum(points**2, axis=1).reshape(-1, 1)
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3].flatten()
    radius = np.sqrt(C[3] + np.sum(center**2))
    return center, radius

# Funzione per il fitting ellittico (approssimazione ellissoide)
def fit_ellipsoid(points):
    # Funzione di fitting ellittico (approssimazione numerica)
    def ellipsoid_model(params, points):
        x0, y0, z0, a, b, c, theta, phi = params
        x, y, z = points.T
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        x_rot = cos_phi * cos_theta * (x - x0) + sin_phi * (y - y0)
        y_rot = -sin_phi * cos_theta * (x - x0) + cos_phi * (y - y0)
        z_rot = sin_theta * (x - x0) + cos_theta * (z - z0)
        return ((x_rot / a) ** 2 + (y_rot / b) ** 2 + (z_rot / c) ** 2 - 1)

    params_init = np.array([0, 0, 0, 1, 1, 1, 0, 0])  # Parametri iniziali
    res = least_squares(ellipsoid_model, params_init, args=(points,))
    return res.x  # Restituisce i parametri dell'ellissoide

# Funzione per il fitting cilindrico (approssimazione cilindro)
def fit_cylinder(points):
    # Questa è una versione semplificata, potremmo usare RANSAC per trovare il cilindro
    pass  # Aggiungere il codice per il fitting cilindrico

# Funzione per scegliere automaticamente la primitiva migliore
def choose_best_fit(points):
    # Esegui fitting sferico
    center, radius = fit_sphere(points)
    rmse_sphere = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    
    # Se la sfera si adatta bene (errore basso), ritorna la sfera
    if rmse_sphere < 0.02:  # Soglia per il fitting sferico (adatta secondo necessità)
        return 'Sphere', center, radius

    # Se la sfera non è abbastanza buona, prova l'ellissoide
    params_ellipsoid = fit_ellipsoid(points)
    # Estrai i parametri dell'ellissoide
    x0, y0, z0, a, b, c, theta, phi = params_ellipsoid
    rmse_ellipsoid = np.sqrt(np.mean((ellipsoid_model(params_ellipsoid, points)) ** 2))

    # Se l'ellissoide si adatta bene
    if rmse_ellipsoid < 0.02:
        return 'Ellipsoid', (x0, y0, z0), (a, b, c)

    # Se né la sfera né l'ellissoide si adattano, prova un cilindro (approssimazione semplificata)
    # Puoi aggiungere il fitting cilindrico qui se necessario
    # altrimenti torna un cilindro di default
    return 'Cylinder', (0, 0, 0), (1, 1, 1)  # Fitting cilindrico fittizio per ora

# === Carica la point cloud della pallina ===
frame = "0000"  # Frame, puoi modificarlo per il tuo caso
dir_path = "ball_pointclouds"
ball_path = f"{dir_path}/ball_frame_{frame}.ply"  # Percorso della point cloud della pallina
pcd_ball = o3d.io.read_point_cloud(ball_path)

# Estrai i punti dalla point cloud della pallina
points_ball = np.asarray(pcd_ball.points)

# Scegli la primitiva migliore
fit_type, fit_center, fit_params = choose_best_fit(points_ball)
print(f"Tipo di fitting: {fit_type}")
print(f"Centro stimato: {fit_center}")
print(f"Parametri di fitting: {fit_params}")

# === Visualizzazione ===
pcd_object = o3d.geometry.PointCloud()
pcd_object.points = o3d.utility.Vector3dVector(points_ball)
o3d.visualization.draw_geometries([pcd_object], window_name="Pallina da tennis")

# Visualizza la primitiva stimata
if fit_type == 'Sphere':
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=fit_params)
    sphere_mesh.translate(fit_center)
    o3d.visualization.draw_geometries([pcd_object, sphere_mesh], window_name="Sfera stimata sulla pallina")
elif fit_type == 'Ellipsoid':
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = fit_params[0] * np.cos(u) * np.sin(v)
    y = fit_params[1] * np.sin(u) * np.sin(v)
    z = fit_params[2] * np.cos(v)

    # Ruotiamo e trasliamo l'ellissoide
    x_rot = x * np.cos(fit_params[7]) * np.cos(fit_params[6]) + y * np.sin(fit_params[7])
    y_rot = -x * np.sin(fit_params[7]) * np.cos(fit_params[6]) + y * np.cos(fit_params[7])
    z_rot = x * np.sin(fit_params[6]) + z * np.cos(fit_params[6])

    # Traslazione dell'ellissoide
    x_rot += fit_center[0]
    y_rot += fit_center[1]
    z_rot += fit_center[2]

    ellipsoid_mesh = o3d.geometry.TriangleMesh()
    ellipsoid_mesh.vertices = o3d.utility.Vector3dVector(np.vstack((x_rot.flatten(), y_rot.flatten(), z_rot.flatten())).T)
    ellipsoid_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([pcd_object, ellipsoid_mesh], window_name="Ellissoide stimato sulla pallina")
